use std::sync::Arc;

use crate::ollama::chat::ChatRequest;

pub enum Response {
    Eos,
    Error(Arc<str>),
    Token(Arc<str>),
}

pub enum Prompt {
    Generate(Arc<str>),
    Chat(ChatRequest),
}
