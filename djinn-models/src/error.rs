use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
    /// The rendered prompt does not leave room to generate even one token.
    #[error(
        "the prompt has {prompt_tokens} tokens, which does not fit the model's context window of {context_length}"
    )]
    ContextLengthExceeded {
        prompt_tokens: usize,
        context_length: usize,
    },
    /// The loaded model has no chat template (it is not instruction-tuned).
    #[error("the loaded model does not support chat")]
    ChatUnsupported,
}
