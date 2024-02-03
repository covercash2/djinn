use std::io::Write;

use candle::{DType, Tensor};
use candle_core::{self as candle, Device};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;
use clap::Parser;
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::mistral::model::{Model, ModelContext, Variant};
use crate::token_output_stream::TokenOutputStream;
use crate::util::hub_load_safetensors;

pub mod model;

#[derive(Parser)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

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

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

pub async fn run(device: Device, args: Args) -> anyhow::Result<()> {
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

    // prep files
    let start = std::time::Instant::now();

    let variant = if args.quantized {
        Variant::Quantized
    } else {
        Variant::Mistral
    };

    let api = Api::new()?;

    let model = Model::load(&api, variant, args.revision, &device, args.use_flash_attn).await?;

    println!("loaded the model in {:?}", start.elapsed());

    let mut model_context = ModelContext::new(model);

    model_context
        .run(
            args.prompt,
            args.seed,
            args.temperature,
            args.top_p,
            args.sample_len,
            args.repeat_penalty,
            args.repeat_last_n,
        )
        .await?;

    Ok(())
}
