use std::{fmt::Display, str::FromStr, sync::Arc};

use anyhow::anyhow;
use ollama_rs::{
    models::{LocalModel, ModelInfo},
    Ollama,
};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::error::Result;

pub mod chat;
pub mod embeddings;
pub mod generate;

pub const DEFAULT_MODEL: &str = "mistral-nemo";
pub const DEFAULT_DOMAIN: &str = "hoss";
pub const DEFAULT_PORT: u16 = 11434;

#[derive(Debug)]
pub struct Client {
    client: Ollama,
}

impl Client {
    pub async fn new(address: &Url) -> anyhow::Result<Self> {
        let (host, port) = match address.origin() {
            url::Origin::Opaque(origin) => Err(anyhow!("can't parse URL: {origin:?}"))?,
            url::Origin::Tuple(scheme, domain, port) => (format!("{scheme}://{domain}"), port),
        };

        let client = Ollama::new(host, port);
        tracing::debug!("testing client connection");
        for model in client.list_local_models().await? {
            tracing::debug!("model loaded: {model:?}");
        }

        Ok(Self { client })
    }

    pub async fn list_local_models(&self) -> Result<Vec<LocalModel>> {
        Ok(self.client.list_local_models().await?)
    }

    pub async fn model_info(&self, model_name: ModelName) -> Result<ModelInfo> {
        Ok(self.client.show_model_info(model_name.to_string()).await?)
    }
}

#[derive(Clone, Debug)]
pub struct ModelName(pub Arc<str>);

impl Default for ModelName {
    fn default() -> Self {
        ModelName(DEFAULT_MODEL.into())
    }
}

impl FromStr for ModelName {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self, Self::Err> {
        Ok(ModelName(s.into()))
    }
}

impl Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.as_ref())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelHost(Url);

impl ModelHost {
    pub fn url(&self) -> &Url {
        &self.0
    }
}

impl Default for ModelHost {
    fn default() -> Self {
        let url: Url = format!("http://{DEFAULT_DOMAIN}:{DEFAULT_PORT}")
            .parse()
            .expect("should be able to parse default URL");
        ModelHost(url)
    }
}

impl FromStr for ModelHost {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self, Self::Err> {
        Ok(ModelHost(s.parse()?))
    }
}

impl Display for ModelHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.as_str())
    }
}
