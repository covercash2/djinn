use std::sync::Arc;

use ollama_rs::models::LocalModel;

use crate::ollama::chat::ChatRequest;

#[derive(Debug, Clone)]
pub enum Response {
    Eos,
    Error(Arc<str>),
    Token(Arc<str>),
    LocalModels(Vec<LocalModel>),
}

pub enum Prompt {
    Generate(Arc<str>),
    Chat(ChatRequest),
    LocalModels,
}
