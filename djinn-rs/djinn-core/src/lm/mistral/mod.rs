use candle_core::{self as candle};
use futures_util::pin_mut;
use futures_util::StreamExt;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::device::Device;
use crate::lm::Lm;
use crate::lm::ModelSource;

use self::config::ModelConfig;
use self::config::ModelRun;
pub use self::config::RunConfig;

pub mod config;

use super::model::ModelContext;
use super::model::ModelContextBuilder;

pub async fn create_new_context(model_config: &ModelConfig) -> anyhow::Result<ModelContext> {
    // prep files
    let start = std::time::Instant::now();

    let device = model_config.device.try_into()?;

    let revision = match &model_config.model_source {
        ModelSource::HuggingFaceHub { revision } => revision,
        ModelSource::Files {
            weight_files: _,
            tokenizer_file: _,
        } => todo!(),
    };

    let variant = model_config.variant;
    let api = Api::new()?;
    let repo_id = variant.hf_repo_id();
    let repo = api.repo(Repo::with_revision(
        repo_id,
        RepoType::Model,
        revision.to_owned(),
    ));

    let weights = variant
        .load_weights(&repo, &device, model_config.flash_attn)
        .await?;

    let tokenizer_file = repo.get("tokenizer.json").await?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

    tracing::info!("loaded the model in {:?}", start.elapsed());

    Ok(ModelContextBuilder::default()
        .model(weights)
        .tokenizer(tokenizer)
        .device(device)
        .build()?)
}

pub async fn run(run: ModelRun) -> anyhow::Result<ModelRun> {
    tracing::info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    tracing::info!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        run.run_config.temperature,
        run.run_config.repeat_penalty,
        run.run_config.repeat_last_n
    );
    let mut model_context = create_new_context(&run.model_config).await?;
    let stream = model_context.run(run.prompt.clone(), run.run_config.clone());

    pin_mut!(stream);

    while let Some(value) = stream.next().await {
        if let Ok(string_token) = value {
            tracing::info!("{string_token}");
        }
    }

    Ok(run)
}

pub async fn run_model(run: ModelRun) -> anyhow::Result<()> {
    tracing::info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    tracing::info!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        run.run_config.temperature,
        run.run_config.repeat_penalty,
        run.run_config.repeat_last_n,
    );
    let mut model_context = create_new_context(&run.model_config).await?;
    let stream = model_context.run(run.prompt.clone(), run.run_config.clone());

    pin_mut!(stream);

    while let Some(value) = stream.next().await {
        if let Ok(string_token) = value {
            tracing::info!("{string_token}");
        }
    }

    Ok(())
}
