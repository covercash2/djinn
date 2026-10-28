use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow};
use async_stream::stream;
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights as QwenWeights;
use derive_builder::Builder;
use djinn_core::lm::{config::RunConfig, model::ModelArchitecture};
use djinn_core::values::{MaxTokens, ModelId, Prompt, TopP};
use hf_hub::api::tokio::ApiRepo;
use tokio_stream::Stream;
use tracing::instrument;

use crate::error::{Error, Result};
use crate::token_output_stream::TokenOutputStream;

use super::LanguageModel;
use super::chat::{ChatInput, ChatStream, ChatTemplate};
use super::generate::{GenParams, StepModel, generate_stream};
use super::reply::reply_stream;

pub enum Model {
    Qwen25Instruct {
        weights: QwenWeights,
        /// From the GGUF metadata (`qwen2.context_length`).
        context_length: usize,
    },
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::Qwen25Instruct { context_length, .. } => f
                .debug_struct("Model::Qwen25Instruct")
                .field("context_length", context_length)
                .finish_non_exhaustive(),
        }
    }
}

impl Model {
    /// Get the End of Sequence token for a given model
    pub fn eos_token(&self) -> &'static str {
        match self {
            Model::Qwen25Instruct { .. } => "<|im_end|>",
        }
    }

    /// The most tokens (prompt plus generated) the model can attend to.
    pub fn context_length(&self) -> usize {
        match self {
            Model::Qwen25Instruct { context_length, .. } => *context_length,
        }
    }
}

pub async fn load_weights(
    architecture: ModelArchitecture,
    repo: &ApiRepo,
    weights_file: &str,
    device: &Device,
) -> anyhow::Result<Model> {
    let files = hf_files(repo, weights_file).await?;

    load_model(architecture, &files, device).await
}

pub async fn hf_files(repo: &ApiRepo, weights_file: &str) -> anyhow::Result<Vec<PathBuf>> {
    Ok(vec![repo.get(weights_file).await?])
}

pub async fn load_model<P: AsRef<Path>>(
    architecture: ModelArchitecture,
    files: &[P],
    device: &Device,
) -> anyhow::Result<Model> {
    match architecture {
        ModelArchitecture::Qwen25Instruct => {
            let filename = files[0].as_ref();
            let mut file = std::fs::File::open(filename)
                .with_context(|| format!("unable to open {}", filename.display()))?;
            let content = gguf_file::Content::read(&mut file)
                .map_err(|e| e.with_path(filename))
                .context("unable to read GGUF file")?;
            let context_length = content
                .metadata
                .get("qwen2.context_length")
                .context("GGUF metadata has no qwen2.context_length")?
                .to_u32()? as usize;
            let weights = QwenWeights::from_gguf(content, &mut file, device)?;
            Ok(Model::Qwen25Instruct {
                weights,
                context_length,
            })
        }
    }
}

impl Model {
    #[instrument(skip(self))]
    fn forward(
        &mut self,
        index: usize,
        tokens: &[u32],
        device: &Device,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Tensor> {
        tracing::trace!("forward pass on index {index}");
        tracing::debug!("tokens {:?}", &tokens);
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let context = &tokens[start_pos..];
        let input = Tensor::new(context, device)?.unsqueeze(0)?;
        tracing::debug!(input_shape = ?input.shape(), start_pos, context_size);
        let logits = match self {
            Model::Qwen25Instruct { weights, .. } => weights.forward(&input, start_pos),
        }
        .inspect_err(|error| {
            tracing::error!(model = ?self, ?error);
        })?;
        tracing::debug!("logits {:?}", logits);
        // Models return (batch, vocab) or (batch, 1, vocab); the batch is always 1.
        let logits = logits.flatten_all()?.to_dtype(DType::F32)?;
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Model::Qwen25Instruct { weights, .. } => weights.clear_kv_cache(),
        }
    }
}

impl StepModel for Model {
    fn step(
        &mut self,
        tokens: &[u32],
        index: usize,
        device: &Device,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Tensor> {
        self.forward(index, tokens, device, repeat_penalty, repeat_last_n)
    }

    fn clear_kv_cache(&mut self) {
        Model::clear_kv_cache(self)
    }

    fn context_length(&self) -> usize {
        Model::context_length(self)
    }
}

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct ModelContext {
    model: Model,
    #[builder(setter(into))]
    tokenizer: TokenOutputStream,
    device: Device,
    /// Identifier reported by `GET /v1/models`.
    #[builder(default = "ModelId::from_static(\"default\")")]
    model_id: ModelId,
    /// Chat prompt format; `None` for models that are not instruction-tuned.
    #[builder(default)]
    chat_template: Option<Arc<dyn ChatTemplate>>,
}

impl ModelContext {
    pub fn run(
        &mut self,
        prompt: Prompt,
        config: RunConfig,
    ) -> impl Stream<Item = Result<String>> + '_ {
        let prompt = prompt.into_string();
        stream! {
            let RunConfig {
                seed,
                temperature,
                top_p,
                sample_len,
                repeat_penalty,
                repeat_last_n,
                ..
            } = config;

            self.tokenizer.clear();

            tracing::debug!("initializing tokenizer");

            let mut tokens = self
                .tokenizer
                .tokenizer()
                .encode(prompt, true)?
                .get_ids()
                .to_vec();

            // Nothing to continue: the model cannot run on zero tokens (CUDA rejects the empty input),
            // so an empty prompt yields an empty completion.
            if tokens.is_empty() {
                return;
            }

            for &t in tokens.iter() {
                if let Some(t) = self.tokenizer.next_token(t)? {
                    yield Ok(t);
                }
            }

            let eos_token = self
                .tokenizer
                .get_token(self.model.eos_token())
                .ok_or(anyhow!("no EOS token found"))?;

            let mut generated_tokens = 0usize;

            let mut logits_processor = LogitsProcessor::new(seed, Some(temperature.get()), top_p.map(TopP::get));

            tracing::debug!("initialized logits");

            let start_gen = std::time::Instant::now();
            tracing::info!("starting generation");
            for index in 0..sample_len.get() {
                let logits =
                    self.model.forward(index, &tokens, &self.device, repeat_penalty, repeat_last_n)?;

                let next_token = logits_processor.sample(&logits)?;
                tokens.push(next_token);
                generated_tokens += 1;

                if next_token == eos_token {
                    break;
                }

                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    yield Ok(t);
                }
            }

            let dt = start_gen.elapsed();
            tracing::info!("finished generation");
            if let Some(rest) = self.tokenizer.decode_rest()? {
                yield Ok(rest);
            }

            self.model.clear_kv_cache();
            tracing::info!(
                "\n{generated_tokens} tokens generated ({:.2} tokens/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }
    }

    /// Generates an assistant reply for a conversation (see [`LanguageModel::chat`]).
    fn chat_reply(&mut self, input: ChatInput) -> Result<ChatStream<'_>> {
        let template = self.chat_template.clone().ok_or(Error::ChatUnsupported)?;

        let prompt = template.render(&input.messages, &input.tools);

        // Stop on the template's end-of-turn markers and on the model's own EOS token.
        let mut stop_tokens: Vec<u32> = template
            .end_of_turn_markers()
            .iter()
            .filter_map(|marker| self.tokenizer.get_token(marker))
            .collect();
        if let Some(eos) = self.tokenizer.get_token(self.model.eos_token())
            && !stop_tokens.contains(&eos)
        {
            stop_tokens.push(eos);
        }

        let defaults = RunConfig::default();
        let params = GenParams {
            max_tokens: input
                .params
                .max_tokens
                .map_or(DEFAULT_CHAT_MAX_TOKENS, MaxTokens::get),
            seed: input.params.seed.unwrap_or(defaults.seed),
            temperature: input.params.temperature.unwrap_or(defaults.temperature),
            top_p: input.params.top_p.or(defaults.top_p),
            repeat_penalty: defaults.repeat_penalty,
            repeat_last_n: defaults.repeat_last_n,
        };

        let events = generate_stream(
            &mut self.model,
            &mut self.tokenizer,
            &self.device,
            prompt,
            params,
            stop_tokens,
        );
        Ok(Box::pin(reply_stream(template, events, input.params.stop)))
    }
}

/// How many tokens a chat reply may use when the request does not say.
const DEFAULT_CHAT_MAX_TOKENS: usize = 1024;

impl LanguageModel for ModelContext {
    fn run(
        &mut self,
        prompt: Prompt,
        config: RunConfig,
    ) -> std::pin::Pin<Box<dyn tokio_stream::Stream<Item = crate::error::Result<String>> + Send + '_>>
    {
        Box::pin(ModelContext::run(self, prompt, config))
    }

    fn model_id(&self) -> ModelId {
        self.model_id.clone()
    }

    fn chat(&mut self, input: ChatInput) -> ChatStream<'_> {
        match self.chat_reply(input) {
            Ok(stream) => stream,
            Err(error) => Box::pin(futures::stream::once(async move { Err(error) })),
        }
    }
}
