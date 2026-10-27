//! Where the agent gets model replies from.
//!
//! The agent loop only knows the [`Backend`] trait. [`OpenAiBackend`] implements it for any server
//! that speaks the OpenAI chat-completions protocol (the djinn server, Ollama, llama.cpp, vLLM);
//! [`ScriptedBackend`] replays canned replies for tests.

use djinn_core::chat::{ChatRequest, FinishReason, ToolCall};
use futures::stream::BoxStream;
use thiserror::Error;

mod openai;
mod scripted;
mod sse;

pub use openai::{OpenAiBackend, DEFAULT_BASE_URL, DEFAULT_MODEL};
pub use scripted::{ScriptedBackend, ScriptedReply};
pub use sse::{SseEvent, SseParser};

/// What a backend produces while a reply is generated.
#[derive(Clone, Debug, PartialEq)]
pub enum BackendEvent {
    /// A piece of the reply text, to show as it arrives.
    Text(String),
    /// The reply is complete. Always the last event of a successful stream.
    Done(ChatResult),
}

/// A complete model reply.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatResult {
    /// The reply text; `None` when the model only made tool calls.
    pub content: Option<String>,
    /// Complete tool calls whose arguments are valid JSON.
    pub tool_calls: Vec<ToolCall>,
    pub finish: FinishReason,
}

/// Why a backend request failed. Each case has a message fit for showing to the user.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BackendError {
    /// The server could not be reached.
    #[error("could not connect to {url}: {message}")]
    Connection { url: String, message: String },
    /// The server answered with an error status.
    #[error("the server returned HTTP {status}: {message}")]
    Status { status: u16, message: String },
    /// The server reported an error inside a stream that had already started.
    #[error("the server reported an error: {0}")]
    Server(String),
    /// The connection ended before the reply was complete.
    #[error("the stream ended before the response was complete")]
    StreamEndedEarly,
    /// The server sent something that is not a valid chat completion.
    #[error("malformed response: {0}")]
    Malformed(String),
}

/// A source of model replies.
pub trait Backend: Send + Sync {
    /// Sends the conversation and streams the reply: [`BackendEvent::Text`] pieces as they arrive,
    /// then a single [`BackendEvent::Done`]. The stream must not be polled after it yields an error.
    fn chat(&self, request: ChatRequest) -> BoxStream<'_, Result<BackendEvent, BackendError>>;
}

impl<T: Backend + ?Sized> Backend for Box<T> {
    fn chat(&self, request: ChatRequest) -> BoxStream<'_, Result<BackendEvent, BackendError>> {
        (**self).chat(request)
    }
}
