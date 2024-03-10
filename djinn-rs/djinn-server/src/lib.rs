use djinn_core::mistral::{config::ModelRun, create_new_context};
pub use server::{Config, HttpServer};
use tracing::instrument;

use crate::server::HttpServerBuilder;

mod server;

#[instrument]
pub async fn run_server(config: Config) -> anyhow::Result<()> {
    let model_path = &config.model_config;
    tracing::debug!("loading model config at {model_path:?}");
    let contents = tokio::fs::read_to_string(model_path).await?;
    let model_config = toml::from_str::<ModelRun>(&contents)?.model_config;
    let model_context = create_new_context(&model_config).await?;

    tracing::debug!("starting server with config: {config:?}");

    let server = HttpServerBuilder::default()
        .config(config)
        .context(model_context)
        .build()?;

    server.start().await
}
