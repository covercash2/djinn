//! XDG Base Directory helpers for djinn.
//!
//! Config files live under `$XDG_CONFIG_HOME/djinn/` (typically `~/.config/djinn/`).
//! Use [`load`] to deserialize any TOML config type from that directory.

use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;

use super::{validate_and_load, Error, Result};

/// Returns the djinn XDG config directory: `$XDG_CONFIG_HOME/djinn/`.
///
/// Typically resolves to `~/.config/djinn/`.
pub fn config_dir() -> Result<PathBuf> {
    let dirs = xdg::BaseDirectories::with_prefix("djinn")?;
    Ok(dirs.get_config_home())
}

/// Returns the path to a named config file within the djinn XDG config directory.
///
/// # Example
/// ```ignore
/// let path = config_file("image-gen.toml")?;
/// // → ~/.config/djinn/image-gen.toml
/// ```
pub fn config_file(name: &str) -> Result<PathBuf> {
    Ok(config_dir()?.join(name))
}

/// Deserializes a TOML config file into `T`, validating against the JSON Schema for `T`.
///
/// Uses `path` when provided; otherwise falls back to `default_filename` inside
/// the djinn XDG config directory.  Returns `T::default()` when the file does
/// not exist, so callers can always proceed to the env-var / CLI merge step.
pub fn load<T>(path: Option<&Path>, default_filename: &str) -> Result<T>
where
    T: schemars::JsonSchema + DeserializeOwned + Default,
{
    let config_path = match path {
        Some(p) => p.to_path_buf(),
        None => config_file(default_filename)?,
    };

    if !config_path.exists() {
        return Ok(T::default());
    }

    let contents = std::fs::read_to_string(&config_path)
        .map_err(|source| Error::Read { path: config_path.clone(), source })?;

    validate_and_load::<T>(&contents, &config_path)
}
