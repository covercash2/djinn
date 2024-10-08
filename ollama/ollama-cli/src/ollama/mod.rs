use std::{fmt::Display, str::FromStr, sync::Arc};

use anyhow::anyhow;
use ollama_rs::Ollama;
use url::Url;

pub mod embeddings;
pub mod generate;

pub const DEFAULT_MODEL: &str = "mistral-nemo";
pub const DEFAULT_DOMAIN: &str = "hoss";
pub const DEFAULT_PORT: u16 = 11434;

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
        for model in client.list_local_models().await? {
            println!("model loaded: {model:?}");
        }

        Ok(Self { client })
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

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ModelName(s.into()))
    }
}

impl Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.as_ref())
    }
}

#[derive(Clone, Debug)]
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

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ModelHost(s.parse()?))
    }
}

impl Display for ModelHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.as_str())
    }
}
