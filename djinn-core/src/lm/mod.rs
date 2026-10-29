//! Language model configuration types.
//!
//! Model execution lives in the `djinn-models` crate; this module only holds the
//! serializable config surface so that it can be shared without the inference stack.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::values::{RepoId, Revision, WeightsFile};

pub mod config;
pub mod model;

/// Where to load the model from,
/// either HuggingFaceHub or from the file system
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelSource {
    HuggingFaceHub {
        revision: Revision,
        /// Hub repository holding the weights. Defaults to the architecture's standard repository.
        #[serde(default)]
        repo_id: Option<RepoId>,
        /// File to download from the repository (for GGUF models). Defaults to the
        /// architecture's standard file.
        #[serde(default)]
        weights_file: Option<WeightsFile>,
        /// Hub repository to take `tokenizer.json` from, when it is not in `repo_id`
        /// (GGUF repositories usually do not ship a usable tokenizer).
        #[serde(default)]
        tokenizer_repo: Option<RepoId>,
        /// Hub revision to take `tokenizer.json` from. Defaults to `main`.
        #[serde(default)]
        tokenizer_revision: Option<Revision>,
    },
    Files {
        weight_files: Vec<PathBuf>,
        tokenizer_file: PathBuf,
    },
}

impl ModelSource {
    /// The Hugging Face Hub at `revision`, with every other setting at its default.
    pub fn hub(revision: Revision) -> Self {
        ModelSource::HuggingFaceHub {
            revision,
            repo_id: None,
            weights_file: None,
            tokenizer_repo: None,
            tokenizer_revision: None,
        }
    }
}
