use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::anyhow;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;
use derive_new::new;
use hf_hub::api::tokio::{Api as HfApi, ApiRepo};
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use crate::token_output_stream::TokenOutputStream;
use crate::util::hub_load_safetensors;

pub enum Variant {
    Mistral,
    Quantized,
}

pub enum Weights {
    Mistral(Mistral),
    QMistral(QMistral),
}

impl Variant {
    pub async fn load_weights(
        &self,
        repo: &ApiRepo,
        device: &Device,
        use_flash_attn: bool,
    ) -> anyhow::Result<Weights> {
        let files = self.hf_files(&repo).await?;
        let config = Config::config_7b_v0_1(use_flash_attn);

        self.load_weight_files(&files, config, device)
    }

    pub fn hf_repo_id(&self) -> String {
        match self {
            Variant::Mistral => "milstralai/Mistral-7B-v0.1".to_string(),
            Variant::Quantized => "lmz/candle-mistral".to_string(),
        }
    }

    pub async fn hf_files(&self, repo: &ApiRepo) -> anyhow::Result<Vec<PathBuf>> {
        match self {
            Variant::Mistral => hub_load_safetensors(repo, "model.safetensors.index.json").await,
            Variant::Quantized => Ok(vec![repo.get("model-q4k.gguf").await?]),
        }
    }

    pub fn load_weight_files<P: AsRef<Path>>(
        &self,
        files: &[P],
        config: Config,
        device: &Device,
    ) -> anyhow::Result<Weights> {
        match self {
            Variant::Mistral => {
                let dtype = if device.is_cuda() {
                    DType::BF16
                } else {
                    DType::F32
                };
                let vb = unsafe { VarBuilder::from_mmaped_safetensors(files, dtype, device)? };
                let weights = Mistral::new(&config, vb)?;
                Ok(Weights::Mistral(weights))
            }
            Variant::Quantized => {
                let filename = &files[0];
                let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                    filename, &device,
                )?;
                let model = QMistral::new(&config, vb)?;
                Ok(Weights::QMistral(model))
            }
        }
    }
}

impl Weights {
    fn forward(
        &mut self,
        index: usize,
        tokens: &[u32],
        device: &Device,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> anyhow::Result<Tensor> {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let context = &tokens[start_pos..];
        let input = Tensor::new(context, device)?.unsqueeze(0)?;
        let logits = match self {
            Weights::Mistral(m) => m.forward(&input, start_pos)?,
            Weights::QMistral(m) => m.forward(&input, start_pos)?,
        };
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };
        Ok(logits)
    }
}

pub struct Model {
    weights: Weights,
    tokenizer: TokenOutputStream,
    device: Device,
}

#[derive(new)]
pub struct ModelContext {
    model: Model,
}

impl ModelContext {
    pub async fn run(
        &mut self,
        prompt: impl Into<String>,
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        sample_len: usize,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> anyhow::Result<()> {
        let prompt = prompt.into();

        self.model.tokenizer.clear();

        let mut tokens = self
            .model
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        for &t in tokens.iter() {
            if let Some(t) = self.model.tokenizer.next_token(t)? {
                print!("{t}");
            }
        }

        let eos_token = self
            .model
            .tokenizer
            .get_token("</s>")
            .ok_or(anyhow!("no EOS token found"))?;

        std::io::stdout().flush()?;
        let mut generated_tokens = 0usize;

        let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);

        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let logits = self.model.weights.forward(
                index,
                &tokens,
                &self.model.device,
                repeat_penalty,
                repeat_last_n,
            )?;

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if next_token == eos_token {
                break;
            }

            if let Some(t) = self.model.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }

        let dt = start_gen.elapsed();
        if let Some(rest) = self
            .model
            .tokenizer
            .decode_rest()
            .map_err(anyhow::Error::msg)?
        {
            print!("{rest}");
        }
        std::io::stdout().flush()?;

        println!(
            "\n{generated_tokens} tokens generated ({:.2} tokens/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

impl Model {
    pub async fn load(
        api: &HfApi,
        variant: Variant,
        revision: String,
        device: &Device,
        use_flash_attn: bool,
    ) -> anyhow::Result<Model> {
        let repo_id = variant.hf_repo_id();
        let repo = api.repo(Repo::with_revision(repo_id, RepoType::Model, revision));

        let tokenizer_file = repo.get("tokenizer.json").await?;
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
        let tokenizer = TokenOutputStream::new(tokenizer);

        let weights = variant.load_weights(&repo, device, use_flash_attn).await?;

        Ok(Model {
            weights,
            tokenizer,
            device: device.clone(),
        })
    }
}
