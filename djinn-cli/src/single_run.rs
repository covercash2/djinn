use std::path::PathBuf;

use clap::Parser;
use djinn_core::device::Device;
use djinn_core::lm::ModelSource;
use djinn_core::lm::config::RunConfig;
use djinn_core::lm::config::{
    DEFAULT_REPEAT_LAST_N, DEFAULT_REPEAT_PENALTY, DEFAULT_SAMPLE_LEN, DEFAULT_SEED,
    DEFAULT_TEMPERATURE, ModelConfig, ModelRun,
};
use djinn_core::lm::model::ModelArchitecture;
use djinn_core::values::{MaxTokens, ModelId, Prompt, Revision, Temperature, TopP};

#[derive(Parser, Clone)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, value_enum)]
    device: Device,
    #[arg(long)]
    prompt: Prompt,
    #[arg(long)]
    model_id: Option<ModelId>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = DEFAULT_REPEAT_PENALTY)]
    repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = DEFAULT_REPEAT_LAST_N)]
    repeat_last_n: usize,
    #[arg(long)]
    revision: Option<Revision>,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = DEFAULT_SAMPLE_LEN)]
    sample_len: MaxTokens,
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: Temperature,
    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<TopP>,
    #[arg(value_enum)]
    variant: ModelArchitecture,

    #[arg(long)]
    tokenizer_file: Option<PathBuf>,
    #[arg(long)]
    weight_files: Option<PathBuf>,
}

impl TryFrom<Args> for ModelRun {
    type Error = anyhow::Error;

    fn try_from(args: Args) -> Result<Self, Self::Error> {
        let prompt = args.prompt.clone();
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
            revision,
            weight_files,
            tokenizer_file,
            ..
        } = value;

        let model_source = if let Some(revision) = revision {
            ModelSource::hub(revision)
        } else if let Some((weight_files, tokenizer_file)) = weight_files.zip(tokenizer_file) {
            ModelSource::Files {
                weight_files: vec![weight_files],
                tokenizer_file,
            }
        } else {
            panic!("unable to find files");
        };

        Ok(ModelConfig {
            variant,
            device: Some(device),
            model_source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(std::iter::once("single-run").chain(args.iter().copied()))
    }

    #[test]
    fn parser_is_valid() {
        use clap::CommandFactory;
        Args::command().debug_assert();
    }

    #[test]
    fn builds_a_model_run_from_the_flags() {
        let args = parse(&[
            "--device",
            "cpu",
            "--prompt",
            "hello",
            "--revision",
            "main",
            "-n",
            "7",
            "qwen2_5_instruct",
        ])
        .unwrap();
        let run: ModelRun = args.try_into().unwrap();

        assert_eq!(run.prompt, "hello");
        assert_eq!(run.run_config.sample_len, MaxTokens::from_const(7));
        assert_eq!(run.model_config.variant, ModelArchitecture::Qwen25Instruct);
        assert_eq!(run.model_config.device, Some(Device::Cpu));
        assert!(matches!(
            run.model_config.model_source,
            ModelSource::HuggingFaceHub { ref revision, .. } if revision.as_str() == "main"
        ));
    }

    #[test]
    fn weight_and_tokenizer_files_are_paths() {
        let args = parse(&[
            "--device",
            "cpu",
            "--prompt",
            "x",
            "--weight-files",
            "w/model.gguf",
            "--tokenizer-file",
            "w/tokenizer.json",
            "qwen2_5_instruct",
        ])
        .unwrap();
        let run: ModelRun = args.try_into().unwrap();
        match run.model_config.model_source {
            ModelSource::Files {
                weight_files,
                tokenizer_file,
            } => {
                assert_eq!(weight_files, vec![PathBuf::from("w/model.gguf")]);
                assert_eq!(tokenizer_file, PathBuf::from("w/tokenizer.json"));
            }
            other => panic!("expected files, got {other:?}"),
        }
    }

    #[test]
    fn out_of_range_sampling_flags_are_usage_errors_naming_the_flag() {
        for (flag, value) in [
            ("--temperature", "3"),
            ("--top-p", "0"),
            ("--sample-len", "0"),
        ] {
            let error = parse(&[
                "--device",
                "cpu",
                "--prompt",
                "x",
                flag,
                value,
                "qwen2_5_instruct",
            ])
            .err()
            .unwrap_or_else(|| panic!("{flag} {value} should be rejected"));
            assert!(error.to_string().contains(flag), "{error}");
        }
    }

    #[test]
    fn the_removed_base_model_is_not_accepted() {
        assert!(
            parse(&[
                "--device",
                "cpu",
                "--prompt",
                "x",
                "--revision",
                "main",
                "q_mistral"
            ])
            .is_err()
        );
    }

    #[test]
    fn flash_attention_is_no_longer_an_option() {
        assert!(
            parse(&[
                "--device",
                "cpu",
                "--prompt",
                "x",
                "--use-flash-attn",
                "qwen2_5_instruct"
            ])
            .is_err()
        );
    }
}
