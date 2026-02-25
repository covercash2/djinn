use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config;

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

const fn default_seed_schema() -> u64 {
    DEFAULT_SEED
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
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelRun {
    pub prompt: String,
    pub model_config: ModelConfig,
    pub run_config: RunConfig,
}

/// Parameters to the model
///
/// When the `clap` feature is enabled this struct also implements
/// [`clap::Args`], so it can be embedded directly in a CLI command.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RunConfig {
    /// Number of tokens to generate.
    #[serde(default = "default_sample_len")]
    #[cfg_attr(feature = "clap", arg(long, default_value_t = DEFAULT_SAMPLE_LEN, env = "DJINN_LM_SAMPLE_LEN"))]
    pub sample_len: usize,
    /// RNG seed for reproducible generation.
    #[serde(default = "default_seed")]
    #[schemars(default = "default_seed_schema")]
    #[cfg_attr(feature = "clap", arg(long, default_value_t = DEFAULT_SEED, env = "DJINN_LM_SEED"))]
    pub seed: u64,
    /// Number of last tokens to consider for the repeat penalty.
    #[serde(default = "default_repeat_last_n")]
    #[cfg_attr(feature = "clap", arg(long, default_value_t = DEFAULT_REPEAT_LAST_N, env = "DJINN_LM_REPEAT_LAST_N"))]
    pub repeat_last_n: usize,
    /// Penalty applied to repeated tokens (1.0 = no penalty).
    #[serde(default = "default_repeat_penalty")]
    #[cfg_attr(feature = "clap", arg(long, default_value_t = DEFAULT_REPEAT_PENALTY, env = "DJINN_LM_REPEAT_PENALTY"))]
    pub repeat_penalty: f32,
    /// Sampling temperature (lower = more deterministic).
    #[serde(default = "default_temperature")]
    #[cfg_attr(feature = "clap", arg(long, default_value_t = DEFAULT_TEMPERATURE, env = "DJINN_LM_TEMPERATURE"))]
    pub temperature: f64,
    /// Nucleus sampling probability cutoff.
    #[serde(default = "default_top_p", skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_LM_TOP_P"))]
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

pub fn load_config(config_path: impl AsRef<Path>) -> config::Result<RunConfig> {
    let path = config_path.as_ref();
    let contents = std::fs::read_to_string(path)
        .map_err(|source| config::Error::Read { path: path.to_owned(), source })?;
    config::validate_and_load::<RunConfig>(&contents, path)
}

/// Configurations that are loaded on initialization of the model.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelConfig {
    pub variant: ModelArchitecture,
    #[serde(default)]
    pub device: Device,
    /// Set true to use flash attention. Only supported on CUDA
    pub flash_attn: bool,
    pub model_source: ModelSource,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_run_config_has_expected_values() {
        let cfg = RunConfig::default();
        assert_eq!(cfg.sample_len, DEFAULT_SAMPLE_LEN);
        assert_eq!(cfg.repeat_last_n, DEFAULT_REPEAT_LAST_N);
        assert_eq!(cfg.repeat_penalty, DEFAULT_REPEAT_PENALTY);
        assert_eq!(cfg.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(cfg.top_p, DEFAULT_TOP_P);
    }

    #[test]
    fn full_template_deserializes_correctly() {
        let template = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../configs/lm/run-config.toml"
        ))
        .expect("configs/lm/run-config.toml template must exist");
        let cfg: RunConfig = toml::from_str(&template).expect("template must parse");

        assert_eq!(cfg.sample_len, DEFAULT_SAMPLE_LEN);
        assert_eq!(cfg.seed, DEFAULT_SEED);
        assert_eq!(cfg.repeat_last_n, DEFAULT_REPEAT_LAST_N);
        assert_eq!(cfg.repeat_penalty, DEFAULT_REPEAT_PENALTY);
        assert_eq!(cfg.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(cfg.top_p, None);
    }

    #[test]
    fn full_template_snapshot() {
        let template = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../configs/lm/run-config.toml"
        ))
        .expect("configs/lm/run-config.toml template must exist");
        let cfg: RunConfig = toml::from_str(&template).expect("template must parse");
        insta::assert_debug_snapshot!(cfg);
    }

    #[test]
    fn partial_toml_uses_serde_defaults() {
        let cfg: RunConfig = toml::from_str("sample_len = 50").unwrap();
        assert_eq!(cfg.sample_len, 50);
        assert_eq!(cfg.repeat_penalty, DEFAULT_REPEAT_PENALTY);
        assert_eq!(cfg.temperature, DEFAULT_TEMPERATURE);
    }

    #[test]
    fn run_config_roundtrips_via_toml() {
        let original = RunConfig::default();
        let serialized = toml::to_string(&original).unwrap();
        let deserialized: RunConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(original.sample_len, deserialized.sample_len);
        assert_eq!(original.repeat_penalty, deserialized.repeat_penalty);
        assert_eq!(original.temperature, deserialized.temperature);
    }
}
