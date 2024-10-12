use ollama_rs::error::OllamaError;
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;

use crate::lm::Response;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    OllamaRs(#[from] OllamaError),

    #[error("error sending Response over channel")]
    SendResponse(#[from] SendError<Response>),

    #[error("got an unexpected response: {0:?}")]
    UnexpectedResponse(Response),
}
