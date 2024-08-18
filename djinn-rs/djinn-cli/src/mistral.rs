use clap::Parser;
use djinn_core::device::Device;
use djinn_core::lm::config::RunConfig;
use djinn_core::lm::config::{
    ModelConfig, ModelRun, DEFAULT_REPEAT_LAST_N, DEFAULT_REPEAT_PENALTY, DEFAULT_SAMPLE_LEN,
    DEFAULT_SEED, DEFAULT_TEMPERATURE,
};
use djinn_core::lm::model::ModelArchitecture;
use djinn_core::lm::ModelSource;

#[derive(Parser, Clone)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, value_enum)]
    device: Device,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    model_id: Option<String>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = DEFAULT_REPEAT_PENALTY)]
    repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = DEFAULT_REPEAT_LAST_N)]
    repeat_last_n: usize,
    #[arg(long)]
    revision: Option<String>,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = DEFAULT_SAMPLE_LEN)]
    sample_len: usize,
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: f64,
    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,
    /// Only compatible with [`Device::Cuda`]
    #[arg(long)]
    use_flash_attn: bool,
    #[arg(value_enum)]
    variant: ModelArchitecture,

    #[arg(long)]
    tokenizer_file: Option<String>,
    #[arg(long)]
    weight_files: Option<String>,
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
