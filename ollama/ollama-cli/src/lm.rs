use std::sync::Arc;

pub enum Response {
    Eos,
    Error(Arc<str>),
    Token(Arc<str>),
}
