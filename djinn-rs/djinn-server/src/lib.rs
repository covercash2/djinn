use djinn_core::lm::config::ModelRun;
use djinn_core::lm::mistral::create_new_context;
pub use server::{Config, HttpServer};
use tokio::sync::Mutex;
use tracing::instrument;

use crate::server::{Context, HttpServerBuilder};

mod error;
mod handlers;
mod server;

pub use error::{Error, Result};

#[instrument]
pub async fn run_server(config: Config) -> anyhow::Result<()> {
    let model_path = &config.model_config;
    tracing::debug!("loading model config at {model_path:?}");
    let contents = tokio::fs::read_to_string(model_path).await?;
    let model_run = toml::from_str::<ModelRun>(&contents)?;
    let model_config = model_run.model_config;
    let model = create_new_context(&model_config).await?;

    let context = Context { model };

    tracing::debug!("starting server with config: {config:?}");

    let server = HttpServerBuilder::default()
        .config(config)
        .context(Mutex::new(context))
        .build()?;

    server.start().await
}
