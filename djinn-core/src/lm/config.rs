use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config;
use crate::values::{MaxTokens, Prompt, Temperature, TopP};

use crate::{
    device::Device,
    lm::{ModelSource, model::ModelArchitecture},
};

pub const DEFAULT_SAMPLE_LEN: MaxTokens = MaxTokens::from_const(100);
pub const DEFAULT_SEED: u64 = 299792458;
pub const DEFAULT_REPEAT_LAST_N: usize = 64;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
pub const DEFAULT_TEMPERATURE: Temperature = Temperature::from_const(1e-7);
pub const DEFAULT_TOP_P: Option<TopP> = None;

const fn default_sample_len() -> MaxTokens {
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
const fn default_temperature() -> Temperature {
    DEFAULT_TEMPERATURE
}
const fn default_top_p() -> Option<TopP> {
    DEFAULT_TOP_P
}

/// The results of a model run
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelRun {
    pub prompt: Prompt,
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
    pub sample_len: MaxTokens,
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
    pub temperature: Temperature,
    /// Nucleus sampling probability cutoff.
    #[serde(default = "default_top_p", skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "clap", arg(long, env = "DJINN_LM_TOP_P"))]
    pub top_p: Option<TopP>,
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
    let contents = std::fs::read_to_string(path).map_err(|source| config::Error::Read {
        path: path.to_owned(),
        source,
    })?;
    config::validate_and_load::<RunConfig>(&contents, path)
}

/// Configurations that are loaded on initialization of the model.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelConfig {
    pub variant: ModelArchitecture,
    /// Device to run on. When omitted, the best device compiled into the build is used
    /// (CUDA or Metal if enabled, otherwise CPU).
    #[serde(default)]
    pub device: Option<Device>,
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
        assert_eq!(cfg.sample_len, MaxTokens::from_const(50));
        assert_eq!(cfg.repeat_penalty, DEFAULT_REPEAT_PENALTY);
        assert_eq!(cfg.temperature, DEFAULT_TEMPERATURE);
    }

    #[test]
    fn every_shipped_model_config_is_valid() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/model");
        let mut checked = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("configs/model exists") {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "toml") {
                let contents = std::fs::read_to_string(&path).unwrap();
                config::validate_and_load::<ModelConfig>(&contents, &path)
                    .unwrap_or_else(|e| panic!("{} is invalid: {e}", path.display()));
                checked.push(path.file_name().unwrap().to_string_lossy().into_owned());
            }
        }
        checked.sort();
        assert!(
            checked.contains(&"qwen2_5_instruct.toml".to_owned())
                && checked.contains(&"qwen2_5_instruct_0_5b.toml".to_owned()),
            "expected the Qwen configs, found {checked:?}"
        );
    }

    #[test]
    fn qwen_config_names_the_gguf_file_and_tokenizer_repo() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../configs/model/qwen2_5_instruct.toml");
        let config: ModelConfig = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(config.variant, ModelArchitecture::Qwen25Instruct);
        let ModelSource::HuggingFaceHub {
            repo_id,
            weights_file,
            tokenizer_repo,
            ..
        } = config.model_source
        else {
            panic!("expected a hub source");
        };
        assert_eq!(
            repo_id.as_ref().map(|r| r.as_str()),
            Some("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
        );
        assert_eq!(
            weights_file.as_ref().map(|r| r.as_str()),
            Some("qwen2.5-1.5b-instruct-q4_k_m.gguf")
        );
        assert_eq!(
            tokenizer_repo.as_ref().map(|r| r.as_str()),
            Some("Qwen/Qwen2.5-1.5B-Instruct")
        );
    }

    #[test]
    fn a_config_naming_the_removed_base_model_is_rejected_listing_the_valid_variants() {
        let contents =
            "variant = \"q_mistral\"\n[model_source.hugging_face_hub]\nrevision = \"main\"\n";
        let error = config::validate_and_load::<ModelConfig>(contents, Path::new("old.toml"))
            .expect_err("q_mistral is no longer a variant");
        let shown = error.to_string();
        assert!(
            shown.contains("qwen2_5_instruct"),
            "should list the valid variants: {shown}"
        );
    }

    #[test]
    fn leftover_flash_attn_keys_in_old_configs_are_ignored() {
        let config: ModelConfig = toml::from_str(
            "variant = \"qwen2_5_instruct\"\nflash_attn = false\n[model_source.hugging_face_hub]\nrevision = \"main\"\n",
        )
        .unwrap();
        assert_eq!(config.variant, ModelArchitecture::Qwen25Instruct);
    }

    #[test]
    fn old_style_hub_source_without_the_new_fields_still_parses() {
        let config: ModelConfig = toml::from_str(
            "variant = \"qwen2_5_instruct\"\n[model_source.hugging_face_hub]\nrevision = \"main\"\n",
        )
        .unwrap();
        assert_eq!(config.device, None);
        assert!(matches!(
            config.model_source,
            ModelSource::HuggingFaceHub {
                repo_id: None,
                weights_file: None,
                ..
            }
        ));
    }

    fn load(source: &str) -> Result<ModelConfig, config::Error> {
        let header = "variant = \"qwen2_5_instruct\"\n[model_source.hugging_face_hub]\nrevision = \"main\"\n";
        config::validate_and_load::<ModelConfig>(&format!("{header}{source}"), Path::new("t.toml"))
    }

    #[test]
    fn a_malformed_repo_id_is_rejected_naming_the_key() {
        let error = load("repo_id = \"not a repo\"\n").unwrap_err().to_string();
        assert!(error.contains("repo_id"), "{error}");
    }

    #[test]
    fn a_weights_file_that_escapes_the_repository_is_rejected_naming_the_key() {
        let error = load("weights_file = \"../../etc/passwd\"\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("weights_file"), "{error}");
    }

    #[test]
    fn a_bad_tokenizer_repo_and_revision_are_rejected() {
        assert!(load("tokenizer_repo = \"a/b/c\"\n").is_err());
        let bad_revision = "variant = \"qwen2_5_instruct\"\n[model_source.hugging_face_hub]\nrevision = \"ma in\"\n";
        let error = config::validate_and_load::<ModelConfig>(bad_revision, Path::new("t.toml"));
        assert!(error.unwrap_err().to_string().contains("revision"));
    }

    #[test]
    fn a_valid_override_is_accepted() {
        assert!(load("repo_id = \"Qwen/Qwen2.5-0.5B-Instruct-GGUF\"\nweights_file = \"qwen2.5-0.5b-instruct-q4_k_m.gguf\"\n").is_ok());
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
