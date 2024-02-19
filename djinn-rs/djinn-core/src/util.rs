use std::path::PathBuf;

use crate::error::Error;
use candle_core::{self as candle, DType};
use hf_hub::api::tokio::ApiError;
use tokio_stream::StreamExt;

/// Loads the safetensors files for a model from the hub based on a json index file.
pub async fn hub_load_safetensors(
    repo: &hf_hub::api::tokio::ApiRepo,
    json_file: &str,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
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

/// CLI args that are common to all models
trait ModelArgs {
    fn dtype(&self) -> anyhow::Result<DType>;
}
