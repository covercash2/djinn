use std::{fs::DirEntry, path::PathBuf, sync::Arc, time::SystemTime};

use ollama_rs::models::LocalModel;

/// A model definition from different sources.
#[derive(Debug, Clone)]
pub enum ModelDefinition {
    /// A model on the Ollama host
    OllamaRemote(LocalModel),
    /// A Modelfile on the local disk
    LocalCache(LocalModelfile),
    /// A model that has been synced between the local disk and the remote Ollama host
    Synced {
        remote: LocalModel,
        local: LocalModelfile,
    },
}

/// A Modelfile saved to the client machine (the machine running this TUI)
#[derive(Debug, Clone)]
pub struct LocalModelfile {
    pub path: PathBuf,
    pub modified: SystemTime,
}

impl TryFrom<PathBuf> for LocalModelfile {
    type Error = crate::error::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        let modified = path
            .metadata()
            .and_then(|metadata| metadata.modified())
            .map_err(|source| crate::error::Error::ReadFile {
                source,
                path: path.clone(),
            })?;

        Ok(LocalModelfile { path, modified })
    }
}

impl TryFrom<DirEntry> for LocalModelfile {
    type Error = crate::error::Error;

    fn try_from(entry: DirEntry) -> Result<Self, Self::Error> {
        let path = entry.path();
        path.try_into()
    }
}
