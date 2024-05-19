use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{model::Variant, Device};

pub const DEFAULT_SAMPLE_LEN: usize = 100;
pub const DEFAULT_SEED: u64 = 299792458;
pub const DEFAULT_REPEAT_LAST_N: usize = 64;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
pub const DEFAULT_TEMPERATURE: f64 = 1e-7;
pub const DEFAULT_TOP_P: Option<f64> = None;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRun {
    pub prompt: String,
    pub model_config: ModelConfig,
    pub run_config: RunConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub sample_len: usize,
    pub seed: u64,
    pub repeat_last_n: usize,
    pub repeat_penalty: f32,
    pub temperature: f64,
    pub top_p: Option<f64>,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            sample_len: DEFAULT_SAMPLE_LEN,
            seed: DEFAULT_SEED,
            repeat_last_n: DEFAULT_REPEAT_LAST_N,
            repeat_penalty: DEFAULT_REPEAT_PENALTY,
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
        }
    }
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
