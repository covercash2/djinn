use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use candle_core::{self as candle, Device as CandleDevice};
use clap::Parser;
use clap::ValueEnum;
use futures_util::pin_mut;
use futures_util::StreamExt;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use serde::Deserialize;
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::mistral::model::{ModelContextBuilder, Variant};

use self::config::ModelConfig;
use self::config::ModelRun;
use self::config::ModelSource;
pub use self::config::RunConfig;
use self::model::ModelContext;

pub mod config;
pub mod model;

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, ValueEnum, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda,
    Metal,
}

impl TryFrom<Device> for CandleDevice {
    type Error = candle_core::Error;

    fn try_from(value: Device) -> Result<Self, Self::Error> {
        match value {
            Device::Cpu => Ok(CandleDevice::Cpu),
            Device::Cuda => CandleDevice::new_cuda(0),
            Device::Metal => CandleDevice::new_metal(0),
        }
    }
}

#[derive(Parser, Clone)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, value_enum)]
    device: Device,
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
    /// Pass the name of the config to save
    #[arg(long)]
    save_config: Option<String>,
}

impl TryFrom<Args> for ModelRun {
    type Error = anyhow::Error;

    fn try_from(args: Args) -> Result<Self, Self::Error> {
        let prompt = args.prompt.to_owned();
        let run_config: RunConfig = args.clone().into();
        let model_config: ModelConfig = args.try_into()?;

        Ok(ModelRun {
            prompt,
            run_config,
            model_config,
        })
    }
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

impl TryFrom<Args> for ModelConfig {
    type Error = anyhow::Error;
    fn try_from(value: Args) -> Result<Self, Self::Error> {
        let Args {
            variant,
            device,
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

        Ok(ModelConfig {
            variant,
            device,
            flash_attn: use_flash_attn,
            model_source,
        })
    }
}

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
    let save_config = args.save_config.clone();
    let run: ModelRun = args.try_into()?;
    let mut model_context = create_new_context(&run.model_config).await?;
    let stream = model_context.run(&run.prompt, run.run_config.clone());

    pin_mut!(stream);

    while let Some(value) = stream.next().await {
        if let Ok(string_token) = value {
            print!("{string_token}");
        }
    }

    if let Some(name) = save_config {
        let contents = toml::to_string(&run)?;
        let path = PathBuf::from(format!("./configs/mistral/{name}.toml"));
        let mut file = File::create(path)?;
        let _ = file.write_all(contents.as_bytes());
    }

    Ok(())
}
