//! A backend that replays canned replies, for tests of everything built on [`Backend`].

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use djinn_core::chat::{ChatRequest, FinishReason, ToolCall};
use futures::stream::{self, BoxStream};
use futures::StreamExt;

use super::{Backend, BackendError, BackendEvent, ChatResult};

/// One canned reply.
pub enum ScriptedReply {
    /// Text streamed in `pieces`, then a complete result.
    Reply {
        pieces: Vec<String>,
        result: ChatResult,
        /// Pause before each piece, to make a slow model.
        delay: Option<Duration>,
    },
    /// The request fails (after `pieces` of text, if any, were streamed).
    Failure {
        pieces: Vec<String>,
        error: BackendError,
    },
    /// The model never answers (for testing cancellation).
    Hang,
}

impl ScriptedReply {
    /// A plain text answer.
    pub fn text(text: &str) -> Self {
        Self::Reply {
            pieces: vec![text.to_owned()],
            result: ChatResult {
                content: Some(text.to_owned()),
                tool_calls: Vec::new(),
                finish: FinishReason::Stop,
            },
            delay: None,
        }
    }

    /// A reply that calls tools, optionally with text first.
    pub fn tool_calls(preamble: Option<&str>, calls: Vec<ToolCall>) -> Self {
        Self::Reply {
            pieces: preamble.map(str::to_owned).into_iter().collect(),
            result: ChatResult {
                content: preamble.map(str::to_owned),
                tool_calls: calls,
                finish: FinishReason::ToolCalls,
            },
            delay: None,
        }
    }

    /// A single call to `name` with JSON `arguments`.
    pub fn call(id: &str, name: &str, arguments: &str) -> Self {
        Self::tool_calls(None, vec![ToolCall::function(id, name, arguments)])
    }

    /// A request that fails immediately.
    pub fn error(error: BackendError) -> Self {
        Self::Failure {
            pieces: Vec::new(),
            error,
        }
    }

    /// A request that streams `text` and then fails.
    pub fn error_after(text: &str, error: BackendError) -> Self {
        Self::Failure {
            pieces: vec![text.to_owned()],
            error,
        }
    }
}

/// Replays [`ScriptedReply`]s in order and records every request it receives.
///
/// Asking for more replies than were scripted is a bug in the test and panics with a clear message.
pub struct ScriptedBackend {
    replies: Mutex<VecDeque<ScriptedReply>>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl ScriptedBackend {
    pub fn new(replies: impl IntoIterator<Item = ScriptedReply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Every request received so far.
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }

    /// How many scripted replies have not been used yet.
    pub fn remaining(&self) -> usize {
        self.replies.lock().unwrap().len()
    }
}

impl Backend for ScriptedBackend {
    fn chat(&self, request: ChatRequest) -> BoxStream<'_, Result<BackendEvent, BackendError>> {
        self.requests.lock().unwrap().push(request);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("the scripted backend ran out of replies");

        match reply {
            ScriptedReply::Hang => Box::pin(stream::pending()),
            ScriptedReply::Failure { pieces, error } => Box::pin(
                stream::iter(pieces.into_iter().map(|p| Ok(BackendEvent::Text(p))))
                    .chain(stream::once(async move { Err(error) })),
            ),
            ScriptedReply::Reply {
                pieces,
                result,
                delay,
            } => {
                let events = pieces
                    .into_iter()
                    .map(|p| Ok(BackendEvent::Text(p)))
                    .chain(std::iter::once(Ok(BackendEvent::Done(result))));
                Box::pin(stream::iter(events).then(move |event| async move {
                    if let Some(delay) = delay {
                        tokio::time::sleep(delay).await;
                    }
                    event
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use djinn_core::chat::ChatMessage;

    use super::*;

    fn request(text: &str) -> ChatRequest {
        ChatRequest {
            messages: vec![ChatMessage::user(text)],
            ..ChatRequest::default()
        }
    }

    async fn drain(
        backend: &ScriptedBackend,
        text: &str,
    ) -> Vec<Result<BackendEvent, BackendError>> {
        backend.chat(request(text)).collect().await
    }

    #[tokio::test]
    async fn replays_replies_in_order_and_records_requests() {
        let backend =
            ScriptedBackend::new([ScriptedReply::text("one"), ScriptedReply::text("two")]);
        let first = drain(&backend, "a").await;
        let second = drain(&backend, "b").await;

        assert_eq!(first[0], Ok(BackendEvent::Text("one".into())));
        assert!(
            matches!(&first[1], Ok(BackendEvent::Done(r)) if r.content.as_deref() == Some("one"))
        );
        assert_eq!(second[0], Ok(BackendEvent::Text("two".into())));
        assert_eq!(backend.requests().len(), 2);
        assert_eq!(
            backend.requests()[1].messages[0].content.as_deref(),
            Some("b")
        );
        assert_eq!(backend.remaining(), 0);
    }

    #[tokio::test]
    async fn a_tool_call_reply_has_no_text_without_a_preamble() {
        let backend = ScriptedBackend::new([ScriptedReply::call("c1", "calculator", "{}")]);
        let events = drain(&backend, "a").await;
        assert_eq!(events.len(), 1);
        let Ok(BackendEvent::Done(result)) = &events[0] else {
            panic!("expected Done")
        };
        assert_eq!(result.tool_calls[0].function.name, "calculator");
        assert_eq!(result.finish, FinishReason::ToolCalls);
    }

    #[tokio::test]
    async fn failures_stream_their_text_first() {
        let backend = ScriptedBackend::new([ScriptedReply::error_after(
            "part",
            BackendError::StreamEndedEarly,
        )]);
        let events = drain(&backend, "a").await;
        assert_eq!(events[0], Ok(BackendEvent::Text("part".into())));
        assert_eq!(events[1], Err(BackendError::StreamEndedEarly));
    }

    #[tokio::test]
    #[should_panic(expected = "ran out of replies")]
    async fn asking_for_more_replies_than_scripted_panics() {
        let backend = ScriptedBackend::new([]);
        let _ = backend.chat(request("a"));
    }

    #[tokio::test]
    async fn a_hanging_reply_never_completes() {
        let backend = ScriptedBackend::new([ScriptedReply::Hang]);
        let mut stream = backend.chat(request("a"));
        let result = tokio::time::timeout(Duration::from_millis(50), stream.next()).await;
        assert!(result.is_err(), "the stream should still be pending");
    }
}
