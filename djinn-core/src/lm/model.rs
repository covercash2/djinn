use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::values::{RepoId, WeightsFile};

/// The variant of the model to be loaded.
///
/// `qwen2_5_instruct` is Qwen2.5-Instruct in quantized GGUF form: an instruct model with native tool
/// calling. The size (0.5B, 1.5B, ...) is chosen by the `repo_id` / `weights_file` in the model source.
///
/// (Variants are deliberately not documented one by one: a doc comment on a variant makes the
/// generated schema a `oneOf`, whose validation errors no longer list the valid names.)
#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
    PartialOrd,
    Ord,
    ValueEnum,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    #[serde(rename = "qwen2_5_instruct")]
    #[value(name = "qwen2_5_instruct")]
    Qwen25Instruct,
}

impl ModelArchitecture {
    /// The Hugging Face Hub repository that hosts this model's weights by default.
    pub fn hf_repo_id(&self) -> RepoId {
        match self {
            ModelArchitecture::Qwen25Instruct => {
                RepoId::from_static("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
            }
        }
    }

    /// The weights file to fetch from the repository by default.
    pub fn default_weights_file(&self) -> WeightsFile {
        match self {
            ModelArchitecture::Qwen25Instruct => {
                WeightsFile::from_static("qwen2.5-1.5b-instruct-q4_k_m.gguf")
            }
        }
    }

    /// The repository to take `tokenizer.json` from by default.
    pub fn default_tokenizer_repo(&self) -> RepoId {
        match self {
            ModelArchitecture::Qwen25Instruct => RepoId::from_static("Qwen/Qwen2.5-1.5B-Instruct"),
        }
    }

    /// The name used for this variant in config files and on the command line.
    pub fn config_name(&self) -> &'static str {
        match self {
            ModelArchitecture::Qwen25Instruct => "qwen2_5_instruct",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_for_qwen() {
        let arch = ModelArchitecture::Qwen25Instruct;
        assert_eq!(
            arch.hf_repo_id().as_str(),
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        );
        assert_eq!(
            arch.default_weights_file().as_str(),
            "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        );
        assert_eq!(
            arch.default_tokenizer_repo().as_str(),
            "Qwen/Qwen2.5-1.5B-Instruct"
        );
    }

    #[test]
    fn serde_name_matches_config_name() {
        let arch = ModelArchitecture::Qwen25Instruct;
        let json = serde_json::to_string(&arch).unwrap();
        assert_eq!(json, format!("\"{}\"", arch.config_name()));
        let back: ModelArchitecture = serde_json::from_str(&json).unwrap();
        assert_eq!(back, arch);
    }

    #[test]
    fn the_removed_base_model_is_not_a_variant() {
        assert!(serde_json::from_str::<ModelArchitecture>("\"q_mistral\"").is_err());
    }
}
