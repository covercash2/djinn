//! Language Models and configurations
use std::path::PathBuf;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::Result;

use self::mistral::RunConfig;

pub mod mistral;
pub mod model;

pub trait Lm {
    // type Config;
    type Weights;

    fn run(
        &mut self,
        prompt: String,
        config: RunConfig,
        model: Self::Weights,
    ) -> impl Stream<Item = Result<String>> + '_;
}

/// Where to load the model from,
/// either HuggingFaceHub or from the file system
#[derive(Clone, Debug, Serialize, Deserialize)]
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
