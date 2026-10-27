// Requires HF Hub model download and GPU inference — excluded from coverage measurement.
#![coverage(off)]

use candle_core::{self as candle};
use futures::pin_mut;
use futures::StreamExt;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use std::sync::Arc;

use djinn_core::lm::config::{ModelConfig, ModelRun};
use djinn_core::lm::model::ModelArchitecture;
use djinn_core::lm::ModelSource;

use super::chat::ChatTemplate;
use super::model::{load_weights, ModelContext, ModelContextBuilder};
use super::qwen::Qwen25Template;

pub async fn create_new_context(model_config: &ModelConfig) -> anyhow::Result<ModelContext> {
    // prep files
    let start = std::time::Instant::now();

    let device = crate::device::to_candle(crate::device::resolve(model_config.device))?;

    let (revision, repo_id, weights_file, tokenizer_repo) = match &model_config.model_source {
        ModelSource::HuggingFaceHub {
            revision,
            repo_id,
            weights_file,
            tokenizer_repo,
        } => (revision, repo_id, weights_file, tokenizer_repo),
        ModelSource::Files {
            weight_files: _,
            tokenizer_file: _,
        } => todo!(),
    };

    let variant = model_config.variant;
    let api = Api::new()?;
    let repo_id = repo_id.clone().unwrap_or_else(|| variant.hf_repo_id());
    let weights_file = weights_file
        .clone()
        .unwrap_or_else(|| variant.default_weights_file().to_owned());
    let repo = api.repo(Repo::with_revision(
        repo_id,
        RepoType::Model,
        revision.to_owned(),
    ));

    let weights = load_weights(variant, &repo, &weights_file, &device).await?;

    // GGUF repositories usually have no usable tokenizer; take it from the original model repository.
    let tokenizer_repo_id = tokenizer_repo
        .clone()
        .unwrap_or_else(|| variant.default_tokenizer_repo());
    let tokenizer_api_repo = api.repo(Repo::with_revision(
        tokenizer_repo_id,
        RepoType::Model,
        revision.to_owned(),
    ));
    let tokenizer_file = tokenizer_api_repo.get("tokenizer.json").await?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

    tracing::info!("loaded the model in {:?}", start.elapsed());

    let model_id = std::path::Path::new(&weights_file)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| weights_file.clone());
    let chat_template: Option<Arc<dyn ChatTemplate>> = match variant {
        ModelArchitecture::Qwen25Instruct => Some(Arc::new(Qwen25Template)),
    };

    Ok(ModelContextBuilder::default()
        .model(weights)
        .tokenizer(tokenizer)
        .device(device)
        .model_id(model_id)
        .chat_template(chat_template)
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
