use std::path::PathBuf;

use modelfile::modelfile::error::ModelfileError;
use ollama_rs::error::OllamaError;
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;

use crate::{
    lm::Response,
    tui::{event::InputMode, ResponseEvent},
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("error reading file {path}: {source}")]
    ReadFile {
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("error opening file at {path}: {source}")]
    OpenFile {
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("error opening file at {path}: {source}")]
    WriteFile {
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("error creating directory at {path}: {source}")]
    CreateDir {
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("error listing entries at {path}: {source}")]
    ReadDir {
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("keymap should have all modes defined by default. missing {0}")]
    MissingKeymap(InputMode),

    #[error("error indexing collection at index {index}: {msg}")]
    BadIndex { index: usize, msg: &'static str },

    #[error("unable to parse view from string: {0}")]
    ViewParse(&'static str),

    #[error(transparent)]
    Modelfile(#[from] ModelfileError),

    #[error("unable to index instruction in Modelfile: {0}")]
    ModelfileIndex(usize),

    #[error("missing instruction in Modelfile: {0}")]
    ModelfileMissing(String),

    #[error(transparent)]
    OllamaRs(#[from] OllamaError),

    #[error("error sending Response over channel")]
    SendResponse(#[from] SendError<Response>),

    #[error("error deserializing TOML: {0}")]
    TomlDe(#[from] toml::de::Error),

    #[error("error Serializing TOML: {0}")]
    TomlSer(#[from] toml::ser::Error),

    #[error("got an unexpected response: {0:?}")]
    UnexpectedResponse(ResponseEvent),
}
