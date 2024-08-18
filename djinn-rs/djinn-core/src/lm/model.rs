use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use async_stream::stream;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::{
    mistral::{Config as MistralConfig, Model as Mistral},
    quantized_mistral::Model as QMistral,
    starcoder2::{Config as StarcoderConfig, Model as Starcoder},
};
use clap::ValueEnum;
use derive_builder::Builder;
use hf_hub::api::tokio::ApiRepo;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;
use tracing::instrument;

use crate::error::Result;
use crate::hf_hub_ext::hub_load_safetensors;
use crate::token_output_stream::TokenOutputStream;

use super::config::RunConfig;

/// The variant of the model to be loaded
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    /// Main Mistral version
    Mistral,
    /// Quantized Mistral
    QMistral,
    DistilBert,
    Starcoder,
}

#[derive(Debug)]
pub enum Model {
    Mistral {
        weights: Mistral,
        config: MistralConfig,
    },
    QMistral {
        weights: QMistral,
        config: MistralConfig,
    },
    Starcoder {
        weights: Starcoder,
        config: StarcoderConfig,
    },
}

// impl Lm for Mistral {
//     // type Config = candle_transformers::models::mistral::Config;
//     type Weights = Mistral;
//
//     fn run(
//         &mut self,
//         prompt: String,
//         config: RunConfig,
//         model: Self::Weights,
//     ) -> impl Stream<Item = Result<String>> + '_ {
//         todo!()
//     }
// }

impl ModelArchitecture {
    pub async fn load_weights(
        &self,
        repo: &ApiRepo,
        device: &Device,
        use_flash_attn: bool,
    ) -> anyhow::Result<Model> {
        let files = self.hf_files(repo).await?;

        self.load_model(&files, repo, device, use_flash_attn).await
    }

    pub fn hf_repo_id(&self) -> String {
        match self {
            ModelArchitecture::Mistral => "milstralai/Mistral-7B-v0.1",
            ModelArchitecture::QMistral => "lmz/candle-mistral",
            ModelArchitecture::DistilBert => "distilbert/distilbert-base-cased-distilled-squad",
            ModelArchitecture::Starcoder => "bigcode/starcoder2-15b",
        }
        .to_string()
    }

    pub async fn hf_files(&self, repo: &ApiRepo) -> anyhow::Result<Vec<PathBuf>> {
        match self {
            ModelArchitecture::Mistral => {
                hub_load_safetensors(repo, "model.safetensors.index.json").await
            }
            ModelArchitecture::QMistral => Ok(vec![repo.get("model-q4k.gguf").await?]),
            ModelArchitecture::DistilBert => todo!(),
            ModelArchitecture::Starcoder => {
                hub_load_safetensors(repo, "model.safetensors.index.json").await
            }
        }
    }

    pub async fn load_model<P: AsRef<Path>>(
        &self,
        files: &[P],
        repo: &ApiRepo,
        device: &Device,
        use_flash_attn: bool,
    ) -> anyhow::Result<Model> {
        match self {
            ModelArchitecture::Mistral => {
                let file = repo.get("config.json").await?;
                let config = serde_json::from_slice(&std::fs::read(file)?)
                    .context("unable to load Mistral config")?;
                let dtype = if device.is_cuda() {
                    DType::BF16
                } else {
                    DType::F32
                };
                let vb = unsafe { VarBuilder::from_mmaped_safetensors(files, dtype, device)? };
                let weights = Mistral::new(&config, vb)?;
                Ok(Model::Mistral { weights, config })
            }
            ModelArchitecture::QMistral => {
                let config = MistralConfig::config_7b_v0_1(use_flash_attn);
                let filename = &files[0];
                let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                    filename, device,
                )?;
                let weights = QMistral::new(&config, vb)?;
                Ok(Model::QMistral { weights, config })
            }
            ModelArchitecture::DistilBert => todo!(),
            ModelArchitecture::Starcoder => {
                let file = repo.get("config.json").await?;
                let config = serde_json::from_slice(&std::fs::read(file)?)
                    .context("unable to load Starcoder config")?;
                let dtype = if device.is_cuda() {
                    DType::BF16
                } else {
                    DType::F32
                };
                let vb = unsafe { VarBuilder::from_mmaped_safetensors(files, dtype, device)? };
                let weights = Starcoder::new(&config, vb)?;
                Ok(Model::Starcoder { weights, config })
            }
        }
    }
}

impl Model {
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
            Model::Mistral { weights, config: _ } => weights.forward(&input, start_pos),
            Model::QMistral { weights, config: _ } => weights.forward(&input, start_pos),
            Model::Starcoder { weights, config: _ } => weights.forward(&input, start_pos),
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
            Model::Mistral { weights, config: _ } => weights.clear_kv_cache(),
            Model::QMistral { weights, config: _ } => weights.clear_kv_cache(),
            Model::Starcoder { weights, config: _ } => weights.clear_kv_cache(),
        }
    }
}

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct ModelContext {
    model: Model,
    #[builder(setter(into))]
    tokenizer: TokenOutputStream,
    device: Device,
}

impl ModelContext {
    pub fn run(
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
