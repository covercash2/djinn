extern crate accelerate_src;

use std::io::Write;

use anyhow::bail;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{generation::LogitsProcessor, models::llama};
use clap::{Parser, ValueEnum};
use hf_hub::{Repo, RepoType};
use llama::{Llama, LlamaConfig};
use tokenizers::Tokenizer;

use crate::util::hub_load_safetensors;

const EOS_TOKEN: &str = "</s>";

#[derive(Parser, Clone)]
pub struct Args {
    prompt: String,
    #[arg(short = 'l', long, default_value_t = 100)]
    sample_len: u64,
    /// the folder name that contains the safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(short = 'w', long)]
    local_weights: Option<String>,
    /// data type. `Dtype::F16` if no value is supplied
    #[arg(short, long)]
    dtype: Option<String>,
    /// the model version. use V1 for "Narsil/amall-7b"
    #[arg(short, long, default_value_t = ModelVersion::V1)]
    model_version: ModelVersion,
    /// which revision of the model to use
    #[arg(short, long, default_value = "main")]
    revision: String,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    disable_flash_attention: bool,
    /// disable key-value cache
    #[arg(long, action = clap::ArgAction::SetTrue)]
    disable_kv_cache: bool,
    #[arg(short, long, default_value_t = 299792458)]
    seed: u64,
    #[arg(short, long)]
    temperature: Option<f64>,
    /// nucleus sampling probability cutoff
    #[arg(long)]
    top_p: Option<f64>,
    /// penalty to be applied for repeating tokens
    #[arg(long)]
    repeat_penalty: Option<f32>,
    /// the context size to consider for the repeat penalty
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

#[derive(ValueEnum, Clone, Copy)]
pub enum ModelVersion {
    V1,
    V2,
}

impl ToString for ModelVersion {
    fn to_string(&self) -> String {
        match self {
            ModelVersion::V1 => "Narsil/amall-7b".to_string(),
            ModelVersion::V2 => "meta-llama/Llama-2-7b-hf".to_string(),
        }
    }
}

pub async fn run(device: Device, args: Args) -> anyhow::Result<()> {
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };

    let api = hf_hub::api::tokio::Api::new()?;
    let model_id = args.model_version.to_string();
    println!("loading the model weights from {model_id}");
    let api = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));

    let tokenizer_filename = match &args.local_weights {
        Some(path) => (path.to_owned() + "tokenizer.json").into(),
        _ => api.get("tokenizer.json").await?,
    };

    let config_filename = match &args.local_weights {
        Some(path) => (path.to_owned() + "config.json").into(),
        _ => api.get("config.json").await?,
    };
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config = config.into_config(!args.disable_flash_attention);

    let filenames = hub_load_safetensors(&api, "model.safetensors.index.json").await?;

    println!("building the model");
    let cache = llama::Cache::new(!args.disable_kv_cache, dtype, &config, &device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, DType::F32, &device)? };

    let model = Llama::load(vb, &cache, &config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);
    let prompt = args.prompt;

    let mut tokens = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    println!("starting inference loop");
    println!("prompt: {}", &prompt);

    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;

    for index in 0..args.sample_len {
        let context_size = if cache.use_kv_cache && index > 0 {
            1
        } else {
            tokens.len()
        };
        let ctx = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctx, &device)?.unsqueeze(0)?;
        let logits: Tensor = model.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        let logits = match args.repeat_penalty {
            None => logits,
            Some(penalty) => {
                let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    penalty,
                    &tokens[start_at..],
                )?
            }
        };
        index_pos += ctx.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        if let Some(text) = tokenizer.id_to_token(next_token) {
            let text = text.replace('_', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
        if Some(next_token) == eos_token_id {
            break;
        }
    }
    let elapsed = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        token_generated as f64 / elapsed.as_secs_f64(),
    );

    Ok(())
}
