//! Language model execution.
use std::pin::Pin;

use djinn_core::lm::config::RunConfig;
use djinn_core::values::{ModelId, Prompt};
use futures::Stream;

use crate::error::{Error, Result};

pub mod chat;
pub(crate) mod generate;
pub mod loader;
pub mod model;
pub mod qwen;
pub(crate) mod reply;

use chat::{ChatInput, ChatStream};

/// Abstraction over any language model that can stream token completions.
///
/// Implementors must be `Send + Sync` so they can live behind `Arc<Mutex<Context>>`.
/// The returned stream must also be `Send` so it can be polled across tasks.
pub trait LanguageModel: Send + Sync {
    /// Continues `prompt` as plain text, with no chat template.
    fn run(
        &mut self,
        prompt: Prompt,
        config: RunConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + '_>>;

    /// Identifier reported to API clients as the loaded model.
    fn model_id(&self) -> ModelId {
        ModelId::from_static("default")
    }

    /// Generates an assistant reply to a conversation. Models that are not instruction-tuned
    /// answer with [`Error::ChatUnsupported`].
    fn chat(&mut self, _input: ChatInput) -> ChatStream<'_> {
        Box::pin(futures::stream::once(async { Err(Error::ChatUnsupported) }))
    }
}
