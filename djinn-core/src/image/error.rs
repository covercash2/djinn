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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_too_long_error_includes_counts() {
        let err = Error::PromptTooLong { len: 100, max: 77 };
        let msg = err.to_string();
        assert!(msg.contains("100"), "message should include the token count");
        assert!(msg.contains("77"), "message should include the max");
    }

    #[test]
    fn missing_token_error_includes_token_name() {
        let err = Error::MissingToken { token: "<|endoftext|>".to_string() };
        let msg = err.to_string();
        assert!(msg.contains("<|endoftext|>"), "message should contain the token");
    }

    #[test]
    fn missing_clip2_config_error_has_message() {
        let err = Error::MissingClip2Config;
        assert!(!err.to_string().is_empty());
    }
}
