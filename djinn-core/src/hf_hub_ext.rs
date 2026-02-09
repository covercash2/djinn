use std::path::PathBuf;

use crate::error::Error;
use candle_core::{self as candle};
use hf_hub::api::tokio::ApiError;
use tokio_stream::StreamExt;

#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("unable to initialize HuggingFace Hub API: {0}")]
    Init(ApiError),

    #[error("unable to get model from HuggingFace Hub: {0}")]
    GetModel(ApiError),
}

/// A wrapper around the HuggingFace Hub API.
pub struct Hub(hf_hub::api::tokio::Api);

impl Hub {
    /// Creates a new Hub instance.
    pub async fn new() -> Result<Self, HubError> {
        let api = hf_hub::api::tokio::Api::new().map_err(HubError::Init)?;
        Ok(Self(api))
    }

    pub async fn get_model_file(
        &self,
        name: impl Into<String>,
        revision: impl Into<String>,
        file: &str,
    ) -> Result<PathBuf, HubError> {
        let repo = self.0.repo(hf_hub::Repo::with_revision(
            name.into(),
            hf_hub::RepoType::Model,
            revision.into(),
        ));

        repo.get(file).await.map_err(HubError::GetModel)
    }

    #[allow(dead_code)] pub async fn get_model_safetensors(
        &self,
        name: String,
        revision: Option<String>,
    ) -> Result<PathBuf, HubError> {
        let repo = match revision {
            Some(revision) => self.0.repo(hf_hub::Repo::with_revision(
                name,
                hf_hub::RepoType::Model,
                revision,
            )),
            None => self
                .0
                .repo(hf_hub::Repo::new(name, hf_hub::RepoType::Model)),
        };
        repo.get("model/model.safetensors")
            .await
            .map_err(HubError::GetModel)
    }
}

/// Loads the safetensors files for a model from the hub based on a json index file.
pub async fn hub_load_safetensors(
    repo: &hf_hub::api::tokio::ApiRepo,
    json_file: &str,
) -> anyhow::Result<Vec<PathBuf>> {
    let json_file = repo.get(json_file).await?;
    let json_fh = std::fs::File::open(json_file.clone())?;
    let json: serde_json::Value = serde_json::from_reader(&json_fh).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => Err(Error::ParameterFileParse {
            path: json_file.to_owned(),
            message: String::from("no weight map found"),
        }),
        Some(serde_json::Value::Object(map)) => Ok(map),
        Some(_) => Err(Error::ParameterFileParse {
            path: json_file,
            message: String::from("weight map in file is not a map"),
        }),
    }?;
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }

    let files: Vec<Result<PathBuf, ApiError>> =
        tokio_stream::iter(std::iter::repeat(repo).zip(&safetensors_files))
            .then(|(repo, file)| repo.get(file))
            .collect::<Vec<Result<PathBuf, ApiError>>>()
            .await;

    let files: Vec<PathBuf> = files.into_iter().collect::<Result<Vec<_>, ApiError>>()?;

    Ok(files)
}
