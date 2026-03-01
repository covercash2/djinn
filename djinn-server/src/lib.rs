#![feature(coverage_attribute)]

use std::sync::Arc;

use djinn_core::image::clip::{Clip, ClipArgs};
use djinn_core::lm::config::ModelConfig;
use djinn_core::lm::mistral::create_new_context;
pub use server::{Config, HttpServer};
use tokio::sync::Mutex;
use tracing::instrument;

use crate::server::{AppState, HttpServerBuilder};

mod clip;
mod complete;
mod error;
mod openapi;
mod server;

pub use error::{Error, Result};

#[coverage(off)] // loads real model weights and HF Hub — not unit testable
#[instrument]
pub async fn run_server(config: Config) -> anyhow::Result<()> {
    let model_path = &config.model_config;
    tracing::debug!("loading model config at {model_path:?}");
    let contents = tokio::fs::read_to_string(model_path).await?;
    let model_config = djinn_core::config::validate_and_load::<ModelConfig>(&contents, model_path)?;

    let model = Arc::new(Mutex::new(
        Box::new(create_new_context(&model_config).await?)
            as Box<dyn djinn_core::lm::LanguageModel>,
    ));

    tracing::debug!("loading CLIP model...");
    let clip = Arc::new(Box::new(
        Clip::new(ClipArgs {
            tokenizer: std::path::PathBuf::new(),
            device: candle_core::Device::Cpu,
        })
        .await
        .map_err(|e| anyhow::anyhow!("failed to initialize CLIP: {e}"))?,
    ) as Box<dyn djinn_core::image::VisionEncoder>);

    tracing::debug!("starting server with config: {config:?}");

    let server = HttpServerBuilder::default()
        .config(config)
        .state(AppState { model, clip })
        .build()?;

    server.start().await
}
