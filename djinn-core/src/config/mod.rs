use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;

pub const DEFAULT_CONFIG_DIR: &str = "./configs";

pub mod xdg;

/// Errors that can occur when loading configuration files.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// XDG base-directory initialisation failed.
    #[error("XDG directory error: {0}")]
    Xdg(#[from] ::xdg::BaseDirectoriesError),

    /// A config file could not be read from disk.
    #[error("failed to read config file {path}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A config file could not be parsed as TOML.
    #[error("failed to parse config file {path}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    /// A config file failed JSON Schema validation.
    #[error("schema validation failed for {path}:\n{message}")]
    Validation { path: PathBuf, message: String },

    /// An unexpected internal error occurred during schema processing.
    #[error("internal schema error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Parse and validate a TOML config string against the JSON Schema derived from `T`.
///
/// Converts the TOML to JSON for schema validation, then deserializes `T` from the
/// original TOML content.  Returns a [`Error::Validation`] with all collected
/// errors when validation fails, or [`Error::Parse`] for syntax errors.
pub fn validate_and_load<T>(content: &str, path: &Path) -> Result<T>
where
    T: schemars::JsonSchema + DeserializeOwned,
{
    // Parse TOML to a generic value (for JSON-Schema validation).
    let toml_value: toml::Value = content
        .parse()
        .map_err(|source| Error::Parse { path: path.to_owned(), source })?;

    // Convert to serde_json::Value — infallible for all TOML primitive types.
    let json_value = serde_json::to_value(&toml_value)
        .map_err(|e| Error::Internal(format!("TOML->JSON conversion failed: {e}")))?;

    // Build schema from the type and validate.
    let schema_json = serde_json::to_value(schemars::schema_for!(T))
        .map_err(|e| Error::Internal(format!("schema serialization failed: {e}")))?;
    let validator = jsonschema::validator_for(&schema_json)
        .map_err(|e| Error::Internal(format!("invalid generated schema: {e}")))?;

    let errors: Vec<String> = validator
        .iter_errors(&json_value)
        .map(|e| format!("  - {e} at {}", e.instance_path))
        .collect();

    if !errors.is_empty() {
        return Err(Error::Validation {
            path: path.to_owned(),
            message: errors.join("\n"),
        });
    }

    // Deserialize from TOML (preserves TOML semantics for edge cases like integers).
    toml::from_str(content).map_err(|source| Error::Parse { path: path.to_owned(), source })
}
