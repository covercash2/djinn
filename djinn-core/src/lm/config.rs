use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    device::Device,
    lm::{model::ModelArchitecture, ModelSource},
};

pub const DEFAULT_SAMPLE_LEN: usize = 100;
pub const DEFAULT_SEED: u64 = 299792458;
pub const DEFAULT_REPEAT_LAST_N: usize = 64;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
pub const DEFAULT_TEMPERATURE: f64 = 1e-7;
pub const DEFAULT_TOP_P: Option<f64> = None;

const fn default_sample_len() -> usize {
    DEFAULT_SAMPLE_LEN
}

fn default_seed() -> u64 {
    cfg_if::cfg_if! {
        if #[cfg(feature = "fixed-seed")] {
            DEFAULT_SEED
        } else {
            random_seed()
        }
    }
}

#[cfg(not(feature = "fixed-seed"))]
fn random_seed() -> u64 {
    use rand::prelude::*;
    use rand::rng;

    rng().random::<u64>()
}

const fn default_repeat_last_n() -> usize {
    DEFAULT_REPEAT_LAST_N
}
const fn default_repeat_penalty() -> f32 {
    DEFAULT_REPEAT_PENALTY
}
const fn default_temperature() -> f64 {
    DEFAULT_TEMPERATURE
}
const fn default_top_p() -> Option<f64> {
    DEFAULT_TOP_P
}

/// The results of a model run
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRun {
    pub prompt: String,
    pub model_config: ModelConfig,
    pub run_config: RunConfig,
}

/// Parameters to the model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    #[serde(default = "default_sample_len")]
    pub sample_len: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_repeat_last_n")]
    pub repeat_last_n: usize,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p", skip_serializing_if = "Option::is_none")]
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

pub fn load_config(config_path: impl AsRef<Path>) -> anyhow::Result<RunConfig> {
    let config_str = std::fs::read_to_string(config_path)?;
    Ok(toml::from_str(&config_str)?)
}

/// Configurations that are loaded on initialization of the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub variant: ModelArchitecture,
    #[serde(default)]
    pub device: Device,
    /// Set true to use flash attention. Only supported on CUDA
    pub flash_attn: bool,
    pub model_source: ModelSource,
}
