//! Language Models and configurations
use std::path::PathBuf;
use std::pin::Pin;

use config::RunConfig;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::Result;

pub mod config;
pub mod mistral;
pub mod model;

/// Abstraction over any language model that can stream token completions.
///
/// Implementors must be `Send + Sync` so they can live behind `Arc<Mutex<Context>>`.
/// The returned stream must also be `Send` so it can be polled across tokio tasks.
pub trait LanguageModel: Send + Sync {
    fn run(
        &mut self,
        prompt: String,
        config: RunConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + '_>>;
}

/// Where to load the model from,
/// either HuggingFaceHub or from the file system
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    HuggingFaceHub {
        revision: String,
    },
    Files {
        weight_files: Vec<PathBuf>,
        tokenizer_file: PathBuf,
    },
}
