use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::siglip::{Config as SigLipConfig, Model as SigLipModel};
use tokenizers::Tokenizer;

use crate::hf_hub_ext::Hub;
use super::clip::ModelFile;
use super::{VisionEncoder, VisionEncoderError, VisionEncoderResult};


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
    pub async fn new(args: SigLipArgs) -> VisionEncoderResult<Self> {
        let hub = Hub::new().await.map_err(VisionEncoderError::InitHub)?;

        let tokenizer_file = if args.tokenizer.exists() {
            args.tokenizer
        } else {
            load_tokenizer(&hub).await?
        };

        let tokenizer = Tokenizer::from_file(tokenizer_file.as_path()).map_err(|source| {
            VisionEncoderError::LoadTokenizer {
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
    pub fn encode_text(&self, text: &str) -> VisionEncoderResult<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(VisionEncoderError::TokenizerEncode)?;

        let mut ids = encoding.get_ids().to_vec();
        ids.resize(SEQ_LEN, PAD_TOKEN_ID);

        let input_ids = Tensor::new(&ids[..], &self.device)?.unsqueeze(0)?;
        Ok(self.model.get_text_features(&input_ids)?)
    }

    /// Loads and preprocesses an image, then returns a feature vector (shape: `[1, hidden_size]`).
    pub fn encode_image(&self, path: &Path) -> VisionEncoderResult<Tensor> {
        let reader = image::ImageReader::open(path).map_err(|source| VisionEncoderError::LoadImage {
            path: path.to_owned(),
            source: image::ImageError::IoError(source),
        })?;
        let img = reader.decode().map_err(|source| VisionEncoderError::LoadImage {
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

impl VisionEncoder for SigLip {
    fn encode_text(&self, text: &str) -> VisionEncoderResult<Tensor> {
        SigLip::encode_text(self, text)
    }

    fn encode_image(&self, path: &Path) -> VisionEncoderResult<Tensor> {
        SigLip::encode_image(self, path)
    }
}

async fn load_tokenizer(hub: &Hub) -> VisionEncoderResult<PathBuf> {
    let mf = ModelFile::siglip_tokenizer();
    hub.get_model_file(mf.name, mf.revision, &mf.file)
        .await
        .map_err(VisionEncoderError::DownloadTokenizer)
}

async fn load_model_weights(hub: &Hub) -> VisionEncoderResult<PathBuf> {
    let mf = ModelFile::siglip_model();
    hub.get_model_file(mf.name, mf.revision, &mf.file)
        .await
        .map_err(VisionEncoderError::DownloadModel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduce the per-channel normalization formula used in [`SigLip::encode_image`]
    /// so we can unit-test it in isolation without loading any model weights.
    fn normalize_pixel(channel_value: u8, mean: f32, std: f32) -> f32 {
        (channel_value as f32 / 255.0 - mean) / std
    }

    #[test]
    fn pixel_value_0_normalizes_to_minus_one() {
        // With mean=0.5 and std=0.5: (0/255 - 0.5) / 0.5 = -1.0
        let result = normalize_pixel(0, SIGLIP_MEAN[0], SIGLIP_STD[0]);
        assert!((result - (-1.0)).abs() < 1e-5, "expected -1.0, got {result}");
    }

    #[test]
    fn pixel_value_255_normalizes_to_plus_one() {
        // With mean=0.5 and std=0.5: (255/255 - 0.5) / 0.5 = 1.0
        let result = normalize_pixel(255, SIGLIP_MEAN[0], SIGLIP_STD[0]);
        assert!((result - 1.0).abs() < 1e-4, "expected 1.0, got {result}");
    }

    #[test]
    fn midrange_pixel_normalizes_near_zero() {
        // mid-range pixel ≈ 0 after SigLIP normalization
        let result = normalize_pixel(127, SIGLIP_MEAN[0], SIGLIP_STD[0]);
        assert!(result.abs() < 0.01, "expected ~0.0, got {result}");
    }

    #[test]
    fn siglip_mean_and_std_are_uniform_across_channels() {
        // SigLIP uses (0.5, 0.5, 0.5) for both mean and std
        assert_eq!(SIGLIP_MEAN, [0.5, 0.5, 0.5]);
        assert_eq!(SIGLIP_STD, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn image_size_and_seq_len_constants_have_expected_values() {
        assert_eq!(IMAGE_SIZE, 224);
        assert_eq!(SEQ_LEN, 64);
    }

    #[test]
    fn pad_token_id_is_one() {
        assert_eq!(PAD_TOKEN_ID, 1);
    }
}
