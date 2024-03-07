use clap::Parser;
use djinn_core::mistral::config::{ModelConfig, ModelRun, ModelSource};
use djinn_core::mistral::model::Variant;
use djinn_core::{device::Device, mistral::RunConfig};
use std::path::Path;

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
    #[arg(long, default_value_t = 1e-7)]
    temperature: f64,
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
