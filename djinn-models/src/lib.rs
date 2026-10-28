//! Model execution for djinn: loading weights, tokenizing, and streaming generation.
//!
//! This is the only workspace crate that links candle. Shared config types live in
//! `djinn-core`, which must stay free of the inference stack.

#[cfg(feature = "mac")]
extern crate accelerate_src;

pub mod device;
mod error;
pub mod holdback;
pub mod lm;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod token_output_stream;

pub use error::{Error, Result};
