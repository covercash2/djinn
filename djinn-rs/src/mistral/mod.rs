use std::path::PathBuf;

use candle_core::{self as candle, Device};
use clap::Parser;
use futures_util::pin_mut;
use futures_util::StreamExt;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::mistral::model::{ModelContextBuilder, Variant};

use self::config::ModelConfig;
use self::config::ModelSource;
pub use self::config::RunConfig;
use self::model::ModelContext;

pub mod config;
pub mod model;

#[derive(Parser, Clone)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 100)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(value_enum)]
    variant: Variant,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

impl From<Args> for RunConfig {
    fn from(value: Args) -> Self {
        let Args {
            seed,
            temperature,
            top_p,
            sample_len,
            repeat_penalty,
            repeat_last_n,
            ..
        } = value;
        RunConfig {
            seed,
            temperature,
            top_p,
            sample_len,
            repeat_penalty,
            repeat_last_n,
        }
    }
}

impl From<Args> for ModelConfig {
    fn from(value: Args) -> Self {
        let Args {
            variant,
            cpu,
            use_flash_attn,
            revision,
            weight_files,
            tokenizer_file,
            ..
        } = value;

        let model_source = if let Some(revision) = revision {
            ModelSource::HuggingFaceHub { revision }
        } else if let Some((weight_files, tokenizer_file)) = weight_files.zip(tokenizer_file) {
            ModelSource::Files {
                weight_files: vec![weight_files.into()],
                tokenizer_file: tokenizer_file.into(),
            }
        } else {
            panic!("unable to find files");
        };

        ModelConfig {
            variant,
            cpu,
            flash_attn: use_flash_attn,
            model_source,
        }
    }
}

pub async fn create_new_context(
    device: Device,
    model_config: ModelConfig,
) -> anyhow::Result<ModelContext> {
    // prep files
    let start = std::time::Instant::now();

    let revision = match model_config.model_source {
        ModelSource::HuggingFaceHub { revision } => revision,
        ModelSource::Files {
            weight_files: _,
            tokenizer_file: _,
        } => todo!(),
    };

    let variant = model_config.variant;
    let api = Api::new()?;
    let repo_id = variant.hf_repo_id();
    let repo = api.repo(Repo::with_revision(repo_id, RepoType::Model, revision));

    let weights = variant
        .load_weights(&repo, &device, model_config.flash_attn)
        .await?;

    let tokenizer_file = repo.get("tokenizer.json").await?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;

    println!("loaded the model in {:?}", start.elapsed());

    Ok(ModelContextBuilder::default()
        .model(weights)
        .tokenizer(tokenizer)
        .device(device)
        .build()?)
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let model_config: ModelConfig = args.clone().into();
    let mut model_context = create_new_context(device, model_config).await?;

    let run_config: RunConfig = args.clone().into();

    let stream = model_context.run(&args.prompt, run_config);

    pin_mut!(stream);

    while let Some(value) = stream.next().await {
        if let Ok(string_token) = value {
            print!("{string_token}");
        }
    }

    Ok(())
}
