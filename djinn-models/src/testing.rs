//! Models for tests in other crates. Enabled by the `testing` feature.

use std::pin::Pin;

use djinn_core::chat::{FinishReason, Usage};
use djinn_core::lm::config::RunConfig;
use djinn_core::values::{ModelId, Prompt};
use futures::{Stream, stream};

use crate::lm::LanguageModel;
use crate::lm::chat::{ChatEvent, ChatInput, ChatOutcome, ChatStream};
use crate::{Error, Result};

/// A model that replays fixed output instead of running inference.
///
/// - [`run`](LanguageModel::run) streams `tokens`.
/// - [`chat`](LanguageModel::chat) streams `chat_text` pieces, then finishes with `outcome`, and
///   records the conversation it was given in `last_chat_input`.
/// - With `fail` set, both fail with an error.
pub struct ScriptedModel {
    pub tokens: Vec<String>,
    pub chat_text: Vec<String>,
    pub outcome: ChatOutcome,
    pub fail: bool,
    pub chat_supported: bool,
    pub last_chat_input: Option<ChatInput>,
}

impl ScriptedModel {
    /// A model whose chat reply is `text`, with finish reason `stop`.
    pub fn replying(text: &str) -> Self {
        Self {
            tokens: Vec::new(),
            chat_text: vec![text.to_owned()],
            outcome: ChatOutcome {
                content: Some(text.to_owned()),
                tool_calls: Vec::new(),
                finish: FinishReason::Stop,
                usage: Usage::new(5, 3),
            },
            fail: false,
            chat_supported: true,
            last_chat_input: None,
        }
    }

    /// A model that streams `tokens` for raw completion and does not support chat.
    pub fn completing(tokens: &[&str]) -> Self {
        Self {
            tokens: tokens.iter().map(|t| t.to_string()).collect(),
            chat_supported: false,
            ..Self::replying("")
        }
    }

    /// A model whose every request fails.
    pub fn failing() -> Self {
        Self {
            fail: true,
            ..Self::replying("")
        }
    }
}

impl LanguageModel for ScriptedModel {
    fn run(
        &mut self,
        _prompt: Prompt,
        _config: RunConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + '_>> {
        if self.fail {
            return Box::pin(stream::once(async {
                Err(Error::Anyhow(anyhow::anyhow!("model error")))
            }));
        }
        Box::pin(stream::iter(self.tokens.clone().into_iter().map(Ok)))
    }

    fn model_id(&self) -> ModelId {
        ModelId::from_static("scripted")
    }

    fn chat(&mut self, input: ChatInput) -> ChatStream<'_> {
        self.last_chat_input = Some(input);
        if !self.chat_supported {
            return Box::pin(stream::once(async { Err(Error::ChatUnsupported) }));
        }
        if self.fail {
            return Box::pin(stream::once(async {
                Err(Error::Anyhow(anyhow::anyhow!("model error")))
            }));
        }
        let events: Vec<Result<ChatEvent>> = self
            .chat_text
            .iter()
            .cloned()
            .map(|text| Ok(ChatEvent::Text(text)))
            .chain(std::iter::once(Ok(ChatEvent::Done(self.outcome.clone()))))
            .collect();
        Box::pin(stream::iter(events))
    }
}
