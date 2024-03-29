use std::path::{Path, PathBuf};

use anyhow::anyhow;
use async_stream::stream;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;
use clap::ValueEnum;
use derive_builder::Builder;
use hf_hub::api::tokio::ApiRepo;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;
use tracing::{instrument, Span};

use super::config::RunConfig;
use crate::error::{Error, Result};
use crate::lm::Lm;
use crate::token_output_stream::TokenOutputStream;
use crate::util::hub_load_safetensors;

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, ValueEnum, Serialize, Deserialize)]
pub enum Variant {
    /// Main Mistral version
    Mistral,
    /// Quantized Mistral
    Quantized,
}

#[derive(Debug)]
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
        let files = self.hf_files(repo).await?;
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
                    filename, device,
                )?;
                let model = QMistral::new(&config, vb)?;
                Ok(Weights::QMistral(model))
            }
        }
    }
}

impl Weights {
    #[instrument(skip(self))]
    fn forward(
        &mut self,
        index: usize,
        tokens: &[u32],
        device: &Device,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Tensor> {
        tracing::trace!("forward pass on index {index}");
        tracing::debug!("tokens {:?}", &tokens);
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let start_pos = tokens.len().saturating_sub(context_size);
        let context = &tokens[start_pos..];
        let input = Tensor::new(context, device)?.unsqueeze(0)?;
        tracing::debug!(input_shape = ?input.shape(), start_pos, context_size);
        let logits = match self {
            Weights::Mistral(m) => m.forward(&input, start_pos),
            Weights::QMistral(m) => { 
                m.forward(&input, start_pos) 
            },
        }
        .inspect_err(|error| {
            tracing::error!(model = ?self, ?error);
        })?;
        tracing::debug!("logits {:?}", logits);
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

    fn clear_kv_cache(&mut self) {
        match self {
            Weights::Mistral(model) => {
                model.clear_kv_cache()
            }
            Weights::QMistral(model) => {
                model.clear_kv_cache()
            }
        }
    }
}

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct ModelContext {
    model: Weights,
    #[builder(setter(into))]
    tokenizer: TokenOutputStream,
    device: Device,
}

impl Lm for ModelContext {
    type Config = RunConfig;

    fn run(
        &mut self,
        prompt: String,
        config: RunConfig,
    ) -> impl Stream<Item = Result<String>> + '_ {
        let prompt = prompt.to_string();
        stream! {
            let RunConfig {
                seed,
                temperature,
                top_p,
                sample_len,
                repeat_penalty,
                repeat_last_n,
                ..
            } = config;

            self.tokenizer.clear();

            tracing::debug!("initializing tokenizer");

            let mut tokens = self
                .tokenizer
                .tokenizer()
                .encode(prompt, true)?
                .get_ids()
                .to_vec();

            for &t in tokens.iter() {
                if let Some(t) = self.tokenizer.next_token(t)? {
                    yield Ok(t);
                }
            }
            let eos_token = self
                .tokenizer
                .get_token("</s>")
                .ok_or(anyhow!("no EOS token found"))?;

            let mut generated_tokens = 0usize;

            let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), top_p);

            tracing::debug!("initialized logits");

            let start_gen = std::time::Instant::now();
            tracing::info!("starting generation");
            for index in 0..sample_len {
                tracing::debug!(tokens.len = tokens.len());
                let logits =
                    self.model.forward(index, &tokens, &self.device, repeat_penalty, repeat_last_n)?;

                let next_token = logits_processor.sample(&logits)?;
                tokens.push(next_token);
                generated_tokens += 1;

                if next_token == eos_token {
                    break;
                }

                if let Some(t) = self.tokenizer.next_token(next_token)? {
                    yield Ok(t);
                }
            }

            let dt = start_gen.elapsed();
            tracing::info!("finished generation");
            if let Some(rest) = self.tokenizer.decode_rest()? {
                yield Ok(rest);
            }

            self.model.clear_kv_cache();
            tracing::info!(
                "\n{generated_tokens} tokens generated ({:.2} tokens/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }
    }
}
