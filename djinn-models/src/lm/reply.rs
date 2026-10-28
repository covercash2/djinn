//! Turns the raw generation stream into a chat reply: text that is safe to show as it arrives,
//! stop strings trimmed, and tool calls extracted.

use std::sync::Arc;

use async_stream::try_stream;
use djinn_core::chat::{FinishReason, Usage};
use djinn_core::values::StopSequence;
use futures::{Stream, StreamExt, pin_mut};

use super::chat::{ChatEvent, ChatOutcome, ChatTemplate};
use super::generate::{GenEvent, GenFinish};
use crate::error::Result;
use crate::holdback::Holdback;

/// Wraps a generation stream into a chat reply stream.
///
/// While the model writes ordinary text it is released as soon as it can no longer turn into a stop
/// string or the start of a tool call. Once a tool call starts, nothing more is shown until
/// generation ends, then the whole call is parsed and delivered in [`ChatOutcome`]. Reaching a stop
/// string ends the reply immediately (the underlying generator is dropped, which stops generation).
pub(crate) fn reply_stream<'a, S>(
    template: Arc<dyn ChatTemplate>,
    events: S,
    stop: Vec<StopSequence>,
) -> impl Stream<Item = Result<ChatEvent>> + Send + 'a
where
    S: Stream<Item = Result<GenEvent>> + Send + 'a,
{
    try_stream! {
        pin_mut!(events);

        let stop: Vec<String> = stop.into_iter().map(StopSequence::into_string).collect();
        let stop_count = stop.len();
        let mut triggers = stop.clone();
        triggers.extend(template.tool_call_start().map(str::to_owned));
        let mut visible = Holdback::new(triggers);
        let mut in_tool_call_stops = Holdback::new(stop);

        let mut in_tool_call = false;
        let mut stopped_by_stop_string = false;
        // Text the client has been shown.
        let mut content = String::new();
        // Text from the start of the tool call onwards, which has been held back.
        let mut tool_text = String::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish = FinishReason::Stop;

        while let Some(event) = events.next().await {
            match event? {
                GenEvent::Started { prompt_tokens: n } => prompt_tokens = n,
                GenEvent::Text { text, completion_tokens: n } => {
                    completion_tokens = n;
                    if in_tool_call {
                        let pushed = in_tool_call_stops.push(&text);
                        tool_text.push_str(&pushed.emit);
                        if pushed.hit.is_some() {
                            stopped_by_stop_string = true;
                            break;
                        }
                        continue;
                    }

                    let pushed = visible.push(&text);
                    if !pushed.emit.is_empty() {
                        content.push_str(&pushed.emit);
                        yield ChatEvent::Text(pushed.emit);
                    }
                    if let Some(hit) = pushed.hit {
                        if hit.trigger < stop_count {
                            stopped_by_stop_string = true;
                            break;
                        }
                        in_tool_call = true;
                        let rest = in_tool_call_stops.push(&hit.rest);
                        tool_text.push_str(&rest.emit);
                        if rest.hit.is_some() {
                            stopped_by_stop_string = true;
                            break;
                        }
                    }
                }
                GenEvent::Done { finish: reason, completion_tokens: n } => {
                    completion_tokens = n;
                    finish = match reason {
                        GenFinish::EndOfTurn => FinishReason::Stop,
                        GenFinish::Length => FinishReason::Length,
                    };
                    break;
                }
            }
        }

        if !stopped_by_stop_string {
            // Natural end: whatever was being held back was not a trigger after all.
            if in_tool_call {
                tool_text.push_str(&in_tool_call_stops.finish());
            } else {
                let rest = visible.finish();
                if !rest.is_empty() {
                    content.push_str(&rest);
                    yield ChatEvent::Text(rest);
                }
            }
        }

        let mut tool_calls = Vec::new();
        let mut final_content = Some(content.clone());
        if in_tool_call {
            let parsed = template.parse_output(&format!("{content}{tool_text}"));
            if parsed.tool_calls.is_empty() {
                // Not a valid call: release the markup that was held back so it is not lost.
                content.push_str(&tool_text);
                final_content = Some(content.clone());
                yield ChatEvent::Text(tool_text);
            } else {
                final_content = parsed.content;
                tool_calls = parsed.tool_calls;
                finish = FinishReason::ToolCalls;
            }
        }

        yield ChatEvent::Done(ChatOutcome {
            content: final_content,
            tool_calls,
            finish,
            usage: Usage::new(prompt_tokens, completion_tokens),
        });
    }
}

#[cfg(test)]
mod tests {
    use futures::{executor::block_on, stream};

    use super::*;
    use crate::error::Error;
    use crate::lm::qwen::Qwen25Template;

    fn text(text: &str, tokens: usize) -> Result<GenEvent> {
        Ok(GenEvent::Text {
            text: text.to_owned(),
            completion_tokens: tokens,
        })
    }

    fn started() -> Result<GenEvent> {
        Ok(GenEvent::Started { prompt_tokens: 7 })
    }

    fn done(finish: GenFinish, tokens: usize) -> Result<GenEvent> {
        Ok(GenEvent::Done {
            finish,
            completion_tokens: tokens,
        })
    }

    fn run(events: Vec<Result<GenEvent>>, stop: &[&str]) -> Result<Vec<ChatEvent>> {
        let stop = stop
            .iter()
            .map(|s| StopSequence::new(*s).unwrap())
            .collect();
        block_on(async {
            reply_stream(Arc::new(Qwen25Template), stream::iter(events), stop)
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect()
        })
    }

    fn shown(events: &[ChatEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                ChatEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    fn outcome(events: &[ChatEvent]) -> &ChatOutcome {
        match events.last() {
            Some(ChatEvent::Done(outcome)) => outcome,
            other => panic!("stream did not end with Done: {other:?}"),
        }
    }

    const CALL: &str = "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"1+1\"}}\n</tool_call>";

    #[test]
    fn plain_text_streams_and_ends_with_stop() {
        let events = run(
            vec![
                started(),
                text("Hi", 1),
                text(" there", 2),
                done(GenFinish::EndOfTurn, 2),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(
            events[..2],
            [
                ChatEvent::Text("Hi".into()),
                ChatEvent::Text(" there".into())
            ]
        );
        let outcome = outcome(&events);
        assert_eq!(outcome.content.as_deref(), Some("Hi there"));
        assert!(outcome.tool_calls.is_empty());
        assert_eq!(outcome.finish, FinishReason::Stop);
        assert_eq!(outcome.usage, Usage::new(7, 2));
    }

    #[test]
    fn hitting_the_token_limit_reports_length() {
        let events = run(
            vec![started(), text("cut", 1), done(GenFinish::Length, 1)],
            &[],
        )
        .unwrap();
        assert_eq!(outcome(&events).finish, FinishReason::Length);
        assert_eq!(outcome(&events).content.as_deref(), Some("cut"));
    }

    #[test]
    fn an_empty_reply_has_empty_content() {
        let events = run(vec![started(), done(GenFinish::EndOfTurn, 0)], &[]).unwrap();
        assert_eq!(outcome(&events).content.as_deref(), Some(""));
        assert_eq!(shown(&events), "");
    }

    #[test]
    fn a_stop_string_trims_the_reply_and_ends_it() {
        let events = run(
            vec![
                started(),
                text("done ", 1),
                text("END", 2),
                text(" ignored", 3),
                done(GenFinish::Length, 3),
            ],
            &["END"],
        )
        .unwrap();
        assert_eq!(shown(&events), "done ");
        assert_eq!(outcome(&events).content.as_deref(), Some("done "));
        assert_eq!(outcome(&events).finish, FinishReason::Stop);
    }

    #[test]
    fn a_stop_string_split_across_tokens_is_never_partially_shown() {
        let events = run(
            vec![
                started(),
                text("a E", 1),
                text("N", 2),
                text("D tail", 3),
                done(GenFinish::Length, 3),
            ],
            &["END"],
        )
        .unwrap();
        assert_eq!(shown(&events), "a ");
        assert_eq!(outcome(&events).content.as_deref(), Some("a "));
    }

    #[test]
    fn a_stop_string_ends_the_underlying_generation_early() {
        // The generator yields an error after the stop string; it must never be polled.
        let events = run(
            vec![started(), text("hi END", 1), Err(Error::ChatUnsupported)],
            &["END"],
        )
        .unwrap();
        assert_eq!(outcome(&events).content.as_deref(), Some("hi "));
    }

    #[test]
    fn held_back_text_that_is_not_a_stop_string_is_released_at_the_end() {
        let events = run(
            vec![started(), text("abEN", 1), done(GenFinish::EndOfTurn, 1)],
            &["END"],
        )
        .unwrap();
        assert_eq!(shown(&events), "abEN");
        assert_eq!(outcome(&events).content.as_deref(), Some("abEN"));
    }

    #[test]
    fn a_tool_call_is_not_streamed_as_text_and_is_delivered_at_the_end() {
        let events = run(
            vec![
                started(),
                text("Let me check.\n", 3),
                text(CALL, 20),
                done(GenFinish::EndOfTurn, 20),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(shown(&events), "Let me check.\n");
        let outcome = outcome(&events);
        assert_eq!(outcome.finish, FinishReason::ToolCalls);
        assert_eq!(outcome.content.as_deref(), Some("Let me check."));
        assert_eq!(outcome.tool_calls.len(), 1);
        assert_eq!(outcome.tool_calls[0].function.name.as_str(), "calculator");
        assert_eq!(
            outcome.tool_calls[0].function.arguments.as_str(),
            r#"{"expression":"1+1"}"#
        );
    }

    #[test]
    fn a_tool_call_only_reply_has_no_content() {
        let events = run(
            vec![started(), text(CALL, 20), done(GenFinish::EndOfTurn, 20)],
            &[],
        )
        .unwrap();
        assert_eq!(shown(&events), "");
        assert_eq!(outcome(&events).content, None);
        assert_eq!(outcome(&events).finish, FinishReason::ToolCalls);
    }

    #[test]
    fn the_tool_call_marker_split_across_tokens_never_leaks() {
        let full = format!("Sure.\n{CALL}");
        // Split the text at every character boundary into two fragments.
        for cut in 0..=full.len() {
            if !full.is_char_boundary(cut) {
                continue;
            }
            let events = run(
                vec![
                    started(),
                    text(&full[..cut], 1),
                    text(&full[cut..], 2),
                    done(GenFinish::EndOfTurn, 2),
                ],
                &[],
            )
            .unwrap();
            assert_eq!(shown(&events), "Sure.\n", "cut at {cut} leaked markup");
            assert_eq!(outcome(&events).tool_calls.len(), 1, "cut at {cut}");
        }
    }

    #[test]
    fn two_tool_calls_are_both_returned() {
        let two = format!("{CALL}\n{CALL}");
        let events = run(
            vec![started(), text(&two, 40), done(GenFinish::EndOfTurn, 40)],
            &[],
        )
        .unwrap();
        assert_eq!(outcome(&events).tool_calls.len(), 2);
    }

    #[test]
    fn an_invalid_tool_call_is_released_as_text_not_dropped() {
        let bad = "Hmm <tool_call>{not json}</tool_call>";
        let events = run(
            vec![started(), text(bad, 9), done(GenFinish::EndOfTurn, 9)],
            &[],
        )
        .unwrap();
        assert_eq!(
            shown(&events),
            bad,
            "streamed text must equal the final content"
        );
        let outcome = outcome(&events);
        assert_eq!(outcome.content.as_deref(), Some(bad));
        assert!(outcome.tool_calls.is_empty());
        assert_eq!(outcome.finish, FinishReason::Stop);
    }

    #[test]
    fn a_tool_call_cut_off_by_the_token_limit_is_plain_text_with_length() {
        let cut = "<tool_call>\n{\"name\": \"calc";
        let events = run(
            vec![started(), text(cut, 8), done(GenFinish::Length, 8)],
            &[],
        )
        .unwrap();
        let outcome = outcome(&events);
        assert!(outcome.tool_calls.is_empty());
        assert_eq!(outcome.finish, FinishReason::Length);
        assert_eq!(outcome.content.as_deref(), Some(cut));
        assert_eq!(shown(&events), cut);
    }

    #[test]
    fn a_stop_string_inside_a_tool_call_ends_generation() {
        // Stop at "}}" cuts the call short, so it cannot be parsed and is shown as text.
        let events = run(
            vec![
                started(),
                text("<tool_call>{\"name\": \"a\", \"arguments\": {}}", 5),
                text(" more", 6),
            ],
            &["}}"],
        )
        .unwrap();
        let outcome = outcome(&events);
        assert!(outcome.tool_calls.is_empty());
        assert_eq!(outcome.finish, FinishReason::Stop);
        assert_eq!(
            outcome.content.as_deref(),
            Some("<tool_call>{\"name\": \"a\", \"arguments\": {")
        );
    }

    #[test]
    fn generator_errors_are_propagated() {
        let result = run(
            vec![
                started(),
                text("hi", 1),
                Err(Error::ContextLengthExceeded {
                    prompt_tokens: 9,
                    context_length: 9,
                }),
            ],
            &[],
        );
        assert!(matches!(result, Err(Error::ContextLengthExceeded { .. })));
    }
}
