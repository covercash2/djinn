use djinn_core::image::clip::{Clip, ClipArgs};
use djinn_core::lm::config::ModelConfig;
use djinn_core::lm::mistral::create_new_context;
pub use server::{Config, HttpServer};
use tokio::sync::Mutex;
use tracing::instrument;

use crate::server::{Context, HttpServerBuilder};

mod clip;
mod complete;
mod error;
mod openapi;
mod server;

pub use error::{Error, Result};

#[instrument]
pub async fn run_server(config: Config) -> anyhow::Result<()> {
    let model_path = &config.model_config;
    tracing::debug!("loading model config at {model_path:?}");
    let contents = tokio::fs::read_to_string(model_path).await?;
    let model_config = toml::from_str::<ModelConfig>(&contents)?;
    let model = create_new_context(&model_config).await?;

    tracing::debug!("loading CLIP model...");
    let clip = Clip::new(ClipArgs {
        tokenizer: std::path::PathBuf::new(),
        device: candle_core::Device::Cpu,
    })
    .await
    .map_err(|e| anyhow::anyhow!("failed to initialize CLIP: {e}"))?;

    let context = Context { model, clip };

    tracing::debug!("starting server with config: {config:?}");

    let server = HttpServerBuilder::default()
        .config(config)
        .context(Mutex::new(context))
        .build()?;

    server.start().await
}
