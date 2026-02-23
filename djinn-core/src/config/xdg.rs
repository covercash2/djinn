//! XDG Base Directory helpers for djinn.
//!
//! Config files live under `$XDG_CONFIG_HOME/djinn/` (typically `~/.config/djinn/`).
//! Use [`load`] to deserialize any TOML config type from that directory.

use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;

use super::{Error, Result};

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

/// Deserializes a TOML config file into `T`.
///
/// Uses `path` when provided; otherwise falls back to `default_filename` inside
/// the djinn XDG config directory.  Returns `T::default()` when the file does
/// not exist, so callers can always proceed to the env-var / CLI merge step.
pub fn load<T>(path: Option<&Path>, default_filename: &str) -> Result<T>
where
    T: DeserializeOwned + Default,
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
    toml::from_str(&contents)
        .map_err(|source| Error::Parse { path: config_path, source })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::io::Write as _;

    #[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
    struct TestConfig {
        value: Option<String>,
        count: Option<u32>,
    }

    #[test]
    fn load_returns_default_when_file_does_not_exist() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.toml");
        let config: TestConfig = load(Some(&path), "fallback.toml").unwrap();
        assert_eq!(config, TestConfig::default());
    }

    #[test]
    fn load_parses_valid_toml_from_explicit_path() {
        let mut tmp_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp_file, r#"value = "hello""#).unwrap();
        writeln!(tmp_file, "count = 42").unwrap();

        let config: TestConfig = load(Some(tmp_file.path()), "fallback.toml").unwrap();
        assert_eq!(config.value.as_deref(), Some("hello"));
        assert_eq!(config.count, Some(42));
    }

    #[test]
    fn load_returns_error_for_invalid_toml() {
        let mut tmp_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp_file, "this is not valid toml = = =").unwrap();

        let result: super::super::Result<TestConfig> = load(Some(tmp_file.path()), "fallback.toml");
        assert!(result.is_err(), "expected a parse error");
    }

    #[test]
    fn config_file_returns_path_under_config_dir() {
        let dir = config_dir().unwrap();
        let file = config_file("my-config.toml").unwrap();
        assert_eq!(file, dir.join("my-config.toml"));
    }
}
