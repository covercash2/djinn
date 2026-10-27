//! The token generation loop for chat, separated from any particular model so it can be tested
//! with a scripted model and a tiny tokenizer (no weights needed).

use async_stream::try_stream;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use futures::Stream;

use crate::error::{Error, Result};
use crate::token_output_stream::TokenOutputStream;

/// A model that can be asked for the logits of the next token.
pub(crate) trait StepModel {
    /// Logits (a one-dimensional tensor over the vocabulary) for the token that follows `tokens`.
    /// `index` is 0 for the first call of a generation and counts up by one per generated token.
    fn step(
        &mut self,
        tokens: &[u32],
        index: usize,
        device: &Device,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Tensor>;

    /// Forget all cached attention state.
    fn clear_kv_cache(&mut self);

    /// The most tokens (prompt plus generated) the model can attend to.
    fn context_length(&self) -> usize;
}

/// Sampling and length settings for one generation.
#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: usize,
    pub seed: u64,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

/// Why generation ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenFinish {
    /// The model produced one of the end-of-turn tokens.
    EndOfTurn,
    /// The token budget (the requested maximum, or the room left in the context) ran out.
    Length,
}

/// What the generator produces.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenEvent {
    /// The prompt was accepted; generation is starting.
    Started { prompt_tokens: usize },
    /// Newly decoded text, with the number of tokens generated so far.
    Text {
        text: String,
        completion_tokens: usize,
    },
    /// Generation ended. `completion_tokens` does not count the end-of-turn token.
    Done {
        finish: GenFinish,
        completion_tokens: usize,
    },
}

/// Generates a reply to `prompt`.
///
/// The prompt is *not* echoed. Every call starts from an empty KV cache, so requests never
/// influence each other. Generation stops at the first token in `stop_tokens` (which is never
/// decoded), when `params.max_tokens` is reached, or when the context window is full. A prompt
/// that does not leave room for a single token is rejected with
/// [`Error::ContextLengthExceeded`] before any generation happens.
pub(crate) fn generate_stream<'a, M: StepModel>(
    model: &'a mut M,
    tokenizer: &'a mut TokenOutputStream,
    device: &'a Device,
    prompt: String,
    params: GenParams,
    stop_tokens: Vec<u32>,
) -> impl Stream<Item = Result<GenEvent>> + 'a {
    try_stream! {
        tokenizer.clear();
        model.clear_kv_cache();

        // The chat template already contains every special token, so none are added here.
        let mut tokens = tokenizer
            .tokenizer()
            .encode(prompt, false)?
            .get_ids()
            .to_vec();
        let prompt_tokens = tokens.len();
        let context_length = model.context_length();
        if prompt_tokens >= context_length {
            Err(Error::ContextLengthExceeded { prompt_tokens, context_length })?;
        }
        yield GenEvent::Started { prompt_tokens };

        let budget = params.max_tokens.min(context_length - prompt_tokens);
        let mut sampler = LogitsProcessor::new(params.seed, Some(params.temperature), params.top_p);
        let mut completion_tokens = 0usize;
        let mut finish = GenFinish::Length;

        for index in 0..budget {
            let logits = model.step(&tokens, index, device, params.repeat_penalty, params.repeat_last_n)?;
            let next = sampler.sample(&logits)?;
            if stop_tokens.contains(&next) {
                finish = GenFinish::EndOfTurn;
                break;
            }
            tokens.push(next);
            completion_tokens += 1;
            if let Some(text) = tokenizer.next_token(next)? {
                yield GenEvent::Text { text, completion_tokens };
            }
        }

        if let Some(text) = tokenizer.decode_rest()? {
            yield GenEvent::Text { text, completion_tokens };
        }
        model.clear_kv_cache();
        yield GenEvent::Done { finish, completion_tokens };
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use futures::{executor::block_on, StreamExt};
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, Tokenizer,
    };

    use super::*;

    pub(crate) const VOCAB: &[&str] = &[
        "<unk>",
        "hello",
        "world",
        "foo",
        "bar",
        "baz",
        "<|im_end|>",
        "<|endoftext|>",
    ];

    pub(crate) fn id(word: &str) -> u32 {
        VOCAB
            .iter()
            .position(|w| *w == word)
            .expect("word in test vocab") as u32
    }

    /// A word-level tokenizer over [`VOCAB`] (whitespace separated), good enough to exercise the
    /// streaming decode without downloading anything.
    pub(crate) fn test_tokenizer() -> TokenOutputStream {
        let vocab = VOCAB
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        TokenOutputStream::new(tokenizer)
    }

    /// Replays a fixed list of next tokens as one-hot logits.
    pub(crate) struct ScriptedStep {
        pub script: Vec<u32>,
        pub context_length: usize,
        pub calls: usize,
        pub cleared: usize,
        pub seen_indices: Vec<usize>,
    }

    impl ScriptedStep {
        pub fn new(script: &[&str], context_length: usize) -> Self {
            Self {
                script: script.iter().map(|w| id(w)).collect(),
                context_length,
                calls: 0,
                cleared: 0,
                seen_indices: Vec::new(),
            }
        }
    }

    impl StepModel for ScriptedStep {
        fn step(
            &mut self,
            _tokens: &[u32],
            index: usize,
            device: &Device,
            _repeat_penalty: f32,
            _repeat_last_n: usize,
        ) -> Result<Tensor> {
            self.seen_indices.push(index);
            let next = self.script[self.calls.min(self.script.len() - 1)];
            self.calls += 1;
            let mut logits = vec![0f32; VOCAB.len()];
            logits[next as usize] = 100.0;
            Ok(Tensor::new(logits.as_slice(), device)?)
        }

        fn clear_kv_cache(&mut self) {
            self.cleared += 1;
        }

        fn context_length(&self) -> usize {
            self.context_length
        }
    }

    pub(crate) fn params(max_tokens: usize) -> GenParams {
        GenParams {
            max_tokens,
            seed: 1,
            temperature: 1e-7,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }

    fn stops() -> Vec<u32> {
        vec![id("<|im_end|>"), id("<|endoftext|>")]
    }

    fn collect(model: &mut ScriptedStep, prompt: &str, params: GenParams) -> Result<Vec<GenEvent>> {
        let mut tokenizer = test_tokenizer();
        let device = Device::Cpu;
        block_on(async {
            generate_stream(
                model,
                &mut tokenizer,
                &device,
                prompt.to_string(),
                params,
                stops(),
            )
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect()
        })
    }

    fn joined_text(events: &[GenEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                GenEvent::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn generates_until_the_end_of_turn_token() {
        let mut model = ScriptedStep::new(&["foo", "bar", "<|im_end|>", "baz"], 100);
        let events = collect(&mut model, "hello world", params(50)).unwrap();

        assert_eq!(events[0], GenEvent::Started { prompt_tokens: 2 });
        assert_eq!(joined_text(&events), "foo bar");
        assert_eq!(
            events.last().unwrap(),
            &GenEvent::Done {
                finish: GenFinish::EndOfTurn,
                completion_tokens: 2
            }
        );
        assert_eq!(
            model.calls, 3,
            "stops reading the script at the end-of-turn token"
        );
    }

    #[test]
    fn the_prompt_is_not_echoed_and_end_of_turn_is_never_shown() {
        let mut model = ScriptedStep::new(&["foo", "<|im_end|>"], 100);
        let events = collect(&mut model, "hello world", params(50)).unwrap();
        let text = joined_text(&events);
        assert_eq!(text, "foo");
        assert!(!text.contains("hello") && !text.contains("im_end"));
    }

    #[test]
    fn both_end_of_turn_tokens_stop_generation() {
        let mut model = ScriptedStep::new(&["foo", "<|endoftext|>"], 100);
        let events = collect(&mut model, "hello", params(50)).unwrap();
        assert!(matches!(
            events.last().unwrap(),
            GenEvent::Done {
                finish: GenFinish::EndOfTurn,
                ..
            }
        ));
    }

    #[test]
    fn stops_at_the_token_limit() {
        let mut model = ScriptedStep::new(&["foo"], 100);
        let events = collect(&mut model, "hello", params(3)).unwrap();
        assert_eq!(joined_text(&events), "foo foo foo");
        assert_eq!(
            events.last().unwrap(),
            &GenEvent::Done {
                finish: GenFinish::Length,
                completion_tokens: 3
            }
        );
    }

    #[test]
    fn token_limit_is_capped_to_the_room_left_in_the_context() {
        // 2 prompt tokens in a window of 5 leaves room for 3, however many were requested.
        let mut model = ScriptedStep::new(&["foo"], 5);
        let events = collect(&mut model, "hello world", params(1000)).unwrap();
        assert_eq!(
            events.last().unwrap(),
            &GenEvent::Done {
                finish: GenFinish::Length,
                completion_tokens: 3
            }
        );
    }

    #[test]
    fn a_prompt_that_fills_the_context_is_rejected_before_generating() {
        let mut model = ScriptedStep::new(&["foo"], 2);
        let error = collect(&mut model, "hello world", params(10)).unwrap_err();
        assert!(matches!(
            error,
            Error::ContextLengthExceeded {
                prompt_tokens: 2,
                context_length: 2
            }
        ));
        assert_eq!(model.calls, 0, "no generation step may run");
    }

    #[test]
    fn a_prompt_one_token_shorter_than_the_context_still_runs() {
        let mut model = ScriptedStep::new(&["foo"], 3);
        let events = collect(&mut model, "hello world", params(10)).unwrap();
        assert_eq!(
            events.last().unwrap(),
            &GenEvent::Done {
                finish: GenFinish::Length,
                completion_tokens: 1
            }
        );
    }

    #[test]
    fn zero_max_tokens_generates_nothing() {
        let mut model = ScriptedStep::new(&["foo"], 100);
        let events = collect(&mut model, "hello", params(0)).unwrap();
        assert_eq!(joined_text(&events), "");
        assert_eq!(model.calls, 0);
    }

    #[test]
    fn the_cache_is_cleared_before_and_after_and_steps_count_from_zero() {
        let mut model = ScriptedStep::new(&["foo", "bar", "<|im_end|>"], 100);
        collect(&mut model, "hello", params(50)).unwrap();
        assert_eq!(model.cleared, 2);
        assert_eq!(model.seen_indices, vec![0, 1, 2]);
    }

    #[test]
    fn identical_requests_give_identical_output() {
        let run = || {
            let mut model = ScriptedStep::new(&["foo", "bar", "<|im_end|>"], 100);
            collect(&mut model, "hello world", params(50)).unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn completion_tokens_are_reported_cumulatively() {
        let mut model = ScriptedStep::new(&["foo", "bar", "baz", "<|im_end|>"], 100);
        let events = collect(&mut model, "hello", params(50)).unwrap();
        let counts: Vec<usize> = events
            .iter()
            .filter_map(|e| match e {
                GenEvent::Text {
                    completion_tokens, ..
                } => Some(*completion_tokens),
                _ => None,
            })
            .collect();
        assert_eq!(counts, vec![1, 2, 3]);
    }
}
