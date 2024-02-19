use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unable to read parameters in file: {path}\n{message}")]
    ParameterFileParse { path: PathBuf, message: String },
}
