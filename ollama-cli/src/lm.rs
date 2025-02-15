use std::sync::Arc;

use ollama_rs::models::{LocalModel, ModelInfo};
use strum::{AsRefStr, EnumMessage};

use crate::ollama::{chat::ChatRequest, ModelName};

#[derive(Debug, Clone, AsRefStr)]
pub enum Response {
    Eos,
    Error(Arc<str>),
    Token(Arc<str>),
    LocalModels(Vec<LocalModel>),
    ModelInfo(ModelInfo),
}

#[derive(Debug, Clone)]
pub enum Prompt {
    Generate(Arc<str>),
    Chat(ChatRequest),
    LocalModels,
    ModelInfo(ModelName),
}
