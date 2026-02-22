use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip::{Config as SigLipConfig, Model as SigLipModel};
use tokenizers::Tokenizer;

use crate::hf_hub_ext::Hub;
use super::clip::{ClipError, ModelFile};

/// Re-export `ClipError` as `SigLipError` for consistent naming at call sites.
pub type SigLipError = ClipError;
pub type SigLipResult<T> = std::result::Result<T, SigLipError>;

pub struct SigLipArgs {
    /// Path to a local tokenizer file; downloads from HF Hub when the path does not exist.
    pub tokenizer: PathBuf,
    pub device: Device,
}

/// SigLIP image size expected by base-patch16-224.
const IMAGE_SIZE: usize = 224;
/// Maximum token sequence length for SigLIP text encoder.
const SEQ_LEN: usize = 64;

/// SigLIP image normalization: mean per channel (R, G, B).
const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
/// SigLIP image normalization: std per channel (R, G, B).
const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// SigLIP pad token id (matches `TextConfig::pad_token_id` default).
const PAD_TOKEN_ID: u32 = 1;

pub struct SigLip {
    tokenizer: Tokenizer,
    model: SigLipModel,
    device: Device,
}

impl SigLip {
    pub async fn new(args: SigLipArgs) -> SigLipResult<Self> {
        let hub = Hub::new().await.map_err(ClipError::InitHub)?;

        let tokenizer_file = if args.tokenizer.exists() {
            args.tokenizer
        } else {
            load_tokenizer(&hub).await?
        };

        let tokenizer = Tokenizer::from_file(tokenizer_file.as_path()).map_err(|source| {
            ClipError::LoadTokenizer {
                path: tokenizer_file.clone(),
                source,
            }
        })?;

        let model_path = load_model_weights(&hub).await?;
        let config = SigLipConfig::base_patch16_224();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &args.device)?
        };
        let model = SigLipModel::new(&config, vb)?;

        Ok(Self {
            tokenizer,
            model,
            device: args.device,
        })
    }

    /// Encodes a text prompt into a feature vector (shape: `[1, hidden_size]`).
    pub fn encode_text(&self, text: &str) -> SigLipResult<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(ClipError::TokenizerEncode)?;

        let mut ids = encoding.get_ids().to_vec();
        ids.resize(SEQ_LEN, PAD_TOKEN_ID);

        let input_ids = Tensor::new(&ids[..], &self.device)?.unsqueeze(0)?;
        Ok(self.model.get_text_features(&input_ids)?)
    }

    /// Loads and preprocesses an image, then returns a feature vector (shape: `[1, hidden_size]`).
    pub fn encode_image(&self, path: &Path) -> SigLipResult<Tensor> {
        let reader = image::ImageReader::open(path).map_err(|source| ClipError::LoadImage {
            path: path.to_owned(),
            source: image::ImageError::IoError(source),
        })?;
        let img = reader.decode().map_err(|source| ClipError::LoadImage {
            path: path.to_owned(),
            source,
        })?;

        let img = img
            .resize_to_fill(
                IMAGE_SIZE as u32,
                IMAGE_SIZE as u32,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8();

        // Normalize each pixel using SigLIP's per-channel mean and std, producing
        // a flat vec in (H, W, C) order that will be permuted to (C, H, W) below.
        let pixels: Vec<f32> = img
            .into_raw()
            .chunks_exact(3)
            .flat_map(|px| {
                [
                    (px[0] as f32 / 255.0 - SIGLIP_MEAN[0]) / SIGLIP_STD[0],
                    (px[1] as f32 / 255.0 - SIGLIP_MEAN[1]) / SIGLIP_STD[1],
                    (px[2] as f32 / 255.0 - SIGLIP_MEAN[2]) / SIGLIP_STD[2],
                ]
            })
            .collect();

        let pixel_values =
            Tensor::from_vec(pixels, (IMAGE_SIZE, IMAGE_SIZE, 3), &self.device)?
                .permute((2, 0, 1))? // HWC → CHW
                .unsqueeze(0)?; // CHW → BCHW

        Ok(self.model.get_image_features(&pixel_values)?)
    }
}

impl super::VisionEncoder for SigLip {
    fn encode_text(&self, text: &str) -> super::VisionEncoderResult<Tensor> {
        SigLip::encode_text(self, text)
    }

    fn encode_image(&self, path: &Path) -> super::VisionEncoderResult<Tensor> {
        SigLip::encode_image(self, path)
    }
}

async fn load_tokenizer(hub: &Hub) -> SigLipResult<PathBuf> {
    let mf = ModelFile::siglip_tokenizer();
    hub.get_model_file(mf.name, mf.revision, &mf.file)
        .await
        .map_err(ClipError::DownloadTokenizer)
}

async fn load_model_weights(hub: &Hub) -> SigLipResult<PathBuf> {
    let mf = ModelFile::siglip_model();
    hub.get_model_file(mf.name, mf.revision, &mf.file)
        .await
        .map_err(ClipError::DownloadModel)
}
