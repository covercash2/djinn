use std::sync::Arc;

use ollama_rs::models::{LocalModel, ModelInfo};

use crate::ollama::{chat::ChatRequest, ModelName};

#[derive(Debug, Clone)]
pub enum Response {
    Eos,
    Error(Arc<str>),
    Token(Arc<str>),
    LocalModels(Vec<LocalModel>),
    ModelInfo(ModelInfo),
}

pub enum Prompt {
    Generate(Arc<str>),
    Chat(ChatRequest),
    LocalModels,
    ModelInfo(ModelName),
}
