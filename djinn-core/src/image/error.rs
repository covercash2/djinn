//! Domain-specific error types for image generation.

use std::path::PathBuf;

/// Errors that can occur during Stable Diffusion image generation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Model file download from HuggingFace Hub failed.
    #[error("model file download failed: {0}")]
    ModelDownload(#[from] hf_hub::api::sync::ApiError),

    /// A tokenizer file could not be loaded from disk.
    #[error("failed to load tokenizer from {path:?}")]
    TokenizerLoad {
        path: PathBuf,
        #[source]
        source: tokenizers::Error,
    },

    /// A tokenizer failed to encode a prompt.
    #[error("failed to tokenize prompt")]
    Tokenize(#[source] tokenizers::Error),

    /// The prompt exceeds the model's maximum token count.
    #[error("prompt is too long: {len} tokens exceeds the model maximum of {max}")]
    PromptTooLong { len: usize, max: usize },

    /// The tokenizer vocabulary is missing an expected special token.
    #[error("tokenizer vocabulary is missing expected token {token:?}")]
    MissingToken { token: String },

    /// SDXL requires a secondary CLIP config but none was present in the SD config.
    #[error("CLIP2 config is required for SDXL but was not found in the SD config")]
    MissingClip2Config,

    /// A candle tensor operation failed.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
