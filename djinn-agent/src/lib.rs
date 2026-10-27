//! A small agent harness: an LLM, a few read-only tools, and the loop between them.
//!
//! The agent talks to any OpenAI-compatible chat server through the [`Backend`](backend::Backend)
//! abstraction. This crate must not depend on the inference stack (candle, `djinn-models`) or on
//! `djinn-server`; `just check` enforces that. It shares only the chat wire types from `djinn-core`.

pub mod agent;
pub mod backend;
pub mod cli;
pub mod repl;
pub mod tools;
