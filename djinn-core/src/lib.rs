//! Shared, dependency-light code for the djinn workspace: config loading and
//! validation, and the serializable config types.
//!
//! This crate must stay free of the inference stack (candle, tokenizers, hf-hub);
//! model execution lives in `djinn-models`.

pub mod chat;
pub mod config;
pub mod device;
pub mod lm;
