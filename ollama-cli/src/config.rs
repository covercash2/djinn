use std::path::{Path, PathBuf};

use derive_more::derive::AsRef;
use modelfile::Modelfile;
use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    fs_ext::{read_file_to_string, save_file, FilesystemExt as _, PathExt},
    model_definition::LocalModelfile,
    ollama::ModelHost,
    tui::event::EventDefinitions,
};

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
    /// The location of Modelfiles that have been edited locally.
    #[serde(default)]
    pub model_cache: ModelCache,
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

/// A place for local [`modelfile`]s
#[derive(Debug, Clone, Serialize, Deserialize, AsRef, derive_more::Display)]
#[display("{path:?}")]
pub struct ModelCache {
    path: PathBuf,
}

impl Default for ModelCache {
    fn default() -> Self {
        let path: PathBuf = base_dirs()
            .expect("unable to get base dirs")
            .get_data_home()
            .join("modelfile");

        Self { path }
    }
}

impl ModelCache {
    /// Pass only the filename. This function will add a `.Modelfile` extension.
    pub fn save(&self, name: &str, modelfile: &Modelfile) -> Result<()> {
        let path = self.path.join(format!("{name}.Modelfile"));
        let contents = modelfile.render();
        save_file(path, contents)?;
        Ok(())
    }

    pub fn load(&self) -> Result<Vec<LocalModelfile>> {
        if !self.path.exists() {
            self.path.create_dir_all()?;
        }
        let paths = self
            .path
            .dir()?
            .flat_map(|dir_entry| {
                dir_entry
                    .and_then(|dir_entry| dir_entry.try_into())
                    .inspect_err(|error| {
                        tracing::warn!(
                            %error,
                            "unable to read directory entry",
                        );
                    })
            })
            .collect();

        Ok(paths)
    }

    /// Create a temporary file.
    ///
    /// this is a way to get around some limitations
    /// of the architecture i'm currently using.
    /// since `AppEvents` don't themselves trigger `AppEvents`
    pub fn stage(&self, modelfile: &Modelfile) -> Result<()> {
        self.save(".temp.Modelfile", &modelfile)
    }

    pub fn get_staged(&self) -> Result<Modelfile> {
        let file = self.path.join(".temp.Modelfile");
        let contents = read_file_to_string(file)?;
        Ok(contents.parse()?)
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
