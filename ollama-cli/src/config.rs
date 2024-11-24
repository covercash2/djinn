use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{fs_ext::read_file_to_string, ollama::ModelHost, tui::event::EventDefinitions};

const APP_NAME: &str = "ollama_tui";
const CONFIG_PATH_VAR: &str = "OLLAMA_TUI_CONFIG_PATH";
const CONFIG_FILE_NAME: &str = "config.toml";
const LOG_FILE_NAME: &str = "tui.log";

#[derive(Debug, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub log_file: LogFile,
    #[serde(default)]
    pub host: ModelHost,
    #[serde(default)]
    pub keymap: EventDefinitions,
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        get_config()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogFile(PathBuf);

impl Default for LogFile {
    fn default() -> Self {
        let path = base_dirs()
            .expect("unable to load base dirs")
            .place_state_file(LOG_FILE_NAME)
            .expect("unable to create default log file");
        LogFile(path)
    }
}

impl AsRef<Path> for LogFile {
    fn as_ref(&self) -> &Path {
        self.0.as_ref()
    }
}

fn get_config() -> anyhow::Result<Config> {
    let path = if let Ok(path) = std::env::var(CONFIG_PATH_VAR) {
        path.into()
    } else {
        base_dirs()?.place_config_file(CONFIG_FILE_NAME)?
    };

    let contents = read_file_to_string(path)?;
    let config = toml::from_str(&contents)?;

    Ok(config)
}

fn base_dirs() -> anyhow::Result<xdg::BaseDirectories> {
    Ok(xdg::BaseDirectories::with_prefix(APP_NAME)?)
}
