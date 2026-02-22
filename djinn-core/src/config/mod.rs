use std::path::PathBuf;

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
}

pub type Result<T> = std::result::Result<T, Error>;
