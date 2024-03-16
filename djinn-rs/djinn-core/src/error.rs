use std::path::PathBuf;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unable to read parameters in file: {path}\n{message}")]
    ParameterFileParse { path: PathBuf, message: String },
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}
