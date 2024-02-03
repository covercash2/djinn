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

use crate::token_output_stream::TokenOutputStream;
use crate::util::hub_load_safetensors;

enum Model {
    Mistral(Mistral),
    Quantized(QMistral),
}

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

    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            if args.quantized {
                "lmz/candle-mistral".to_string()
            } else {
                "mistralai/Mistral-7B-v0.1".to_string()
            }
        }
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json").await?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            if args.quantized {
                vec![repo.get("model-q4k.gguf").await?]
            } else {
                hub_load_safetensors(&repo, "model.safetensors.index.json").await?
            }
        }
    };

    println!("retrieved the files in {:?}", start.elapsed());
    // finished prepping files

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    // load files
    let start = std::time::Instant::now();
    let config = Config::config_7b_v0_1(args.use_flash_attn);
    let (mut model, device) = if args.quantized {
        let filename = &filenames[0];
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename, &device)?;
        let model = QMistral::new(&config, vb)?;
        (Model::Quantized(model), device)
    } else {
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = Mistral::new(&config, vb)?;
        (Model::Mistral(model), device)
    };

    println!("loaded the model in {:?}", start.elapsed());

    tokenizer.clear();
    let mut tokens = tokenizer
        .tokenizer()
        .encode(args.prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    for &t in tokens.iter() {
        if let Some(t) = tokenizer.next_token(t)? {
            print!("{t}");
        }
    }
    std::io::stdout().flush()?;

    let mut generated_tokens = 0usize;
    let eos_token = match tokenizer.get_token("</s>") {
        Some(token) => token,
        None => anyhow::bail!("cannot find the </s> token"),
    };

    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);

    let start_gen = std::time::Instant::now();
    for index in 0..args.sample_len {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let context = &tokens[start_pos..];
        let input = Tensor::new(context, &device)?.unsqueeze(0)?;
        let logits = match &mut model {
            Model::Mistral(m) => m.forward(&input, start_pos)?,
            Model::Quantized(m) => m.forward(&input, start_pos)?,
        };
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;

        if next_token == eos_token {
            break;
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    let dt = start_gen.elapsed();
    if let Some(rest) = tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    println!(
        "\n{generated_tokens} tokens generated ({:.2} tokens/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
