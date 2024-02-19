use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{model::Variant, Device};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRun {
    pub prompt: String,
    pub model_config: ModelConfig,
    pub run_config: RunConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub seed: u64,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub sample_len: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

pub fn load_config_named(config_path: impl AsRef<Path>) -> anyhow::Result<RunConfig> {
    let config_str = std::fs::read_to_string(config_path)?;
    Ok(toml::from_str(&config_str)?)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub variant: Variant,
    pub device: Device,
    /// Set true to use flash attention. Only supported on CUDA
    pub flash_attn: bool,
    pub model_source: ModelSource,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelSource {
    HuggingFaceHub {
        revision: String,
    },
    Files {
        weight_files: Vec<PathBuf>,
        tokenizer_file: PathBuf,
    },
}
