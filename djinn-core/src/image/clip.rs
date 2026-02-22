use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{ClipConfig, ClipModel};
use tokenizers::Tokenizer;

use crate::hf_hub_ext::{Hub, HubError};

pub type ClipResult<T> = std::result::Result<T, ClipError>;

#[derive(Debug, thiserror::Error)]
pub enum ClipError {
    #[error("unable to load hub: {0}")]
    InitHub(HubError),

    #[error("couldn't download tokenizer: {0}")]
    DownloadTokenizer(HubError),

    #[error("couldn't load tokenizer from file {path}: {source}")]
    LoadTokenizer {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("unable to encode text: {0}")]
    TokenizerEncode(tokenizers::Error),

    #[error("missing pad token in tokenizer")]
    MissingPadToken,

    #[error("couldn't download model weights: {0}")]
    DownloadModel(HubError),

    #[error("failed to load image from {path}: {source}")]
    LoadImage {
        path: PathBuf,
        source: image::ImageError,
    },

    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub struct ModelFile {
    pub name: String,
    pub revision: String,
    pub file: String,
}

impl ModelFile {
    pub fn new(
        name: impl Into<String>,
        revision: impl Into<String>,
        file: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            revision: revision.into(),
            file: file.into(),
        }
    }

    pub fn clip_tokenizer() -> Self {
        Self::new(
            "openai/clip-vit-base-patch32",
            "refs/pr/15",
            "tokenizer.json",
        )
    }

    pub fn clip_model() -> Self {
        Self::new(
            "openai/clip-vit-base-patch32",
            "refs/pr/15",
            "model.safetensors",
        )
    }

    pub fn siglip_tokenizer() -> Self {
        Self::new(
            "google/siglip-base-patch16-224",
            "main",
            "tokenizer.json",
        )
    }

    pub fn siglip_model() -> Self {
        Self::new(
            "google/siglip-base-patch16-224",
            "main",
            "model.safetensors",
        )
    }
}

pub struct ClipArgs {
    pub tokenizer: PathBuf,
    pub device: Device,
}

/// CLIP image size expected by ViT-B/32.
const IMAGE_SIZE: usize = 224;
/// Maximum token sequence length for CLIP text encoder.
const SEQ_LEN: usize = 77;

/// CLIP image normalization: mean per channel (R, G, B).
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
/// CLIP image normalization: std per channel (R, G, B).
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub struct Clip {
    tokenizer: Tokenizer,
    model: ClipModel,
    device: Device,
}

impl Clip {
    pub async fn new(args: ClipArgs) -> ClipResult<Self> {
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
        let config = ClipConfig::vit_base_patch32();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &args.device)?
        };
        let model = ClipModel::new(vb, &config)?;

        Ok(Self {
            tokenizer,
            model,
            device: args.device,
        })
    }

    /// Encodes a text prompt into a normalized feature vector (shape: `[1, projection_dim]`).
    pub fn encode_text(&self, text: &str) -> ClipResult<Tensor> {
        let pad_id = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or(ClipError::MissingPadToken)?;

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(ClipError::TokenizerEncode)?;

        let mut ids = encoding.get_ids().to_vec();
        ids.resize(SEQ_LEN, pad_id);

        let input_ids = Tensor::new(&ids[..], &self.device)?.unsqueeze(0)?;
        Ok(self.model.get_text_features(&input_ids)?)
    }

    /// Loads and preprocesses an image, then returns a normalized feature vector (shape: `[1, projection_dim]`).
    pub fn encode_image(&self, path: &Path) -> ClipResult<Tensor> {
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

        // Normalize each pixel using CLIP's per-channel mean and std, producing
        // a flat vec in (H, W, C) order that will be permuted to (C, H, W) below.
        let pixels: Vec<f32> = img
            .into_raw()
            .chunks_exact(3)
            .flat_map(|px| {
                [
                    (px[0] as f32 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0],
                    (px[1] as f32 / 255.0 - CLIP_MEAN[1]) / CLIP_STD[1],
                    (px[2] as f32 / 255.0 - CLIP_MEAN[2]) / CLIP_STD[2],
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

impl super::VisionEncoder for Clip {
    fn encode_text(&self, text: &str) -> super::VisionEncoderResult<Tensor> {
        Clip::encode_text(self, text)
    }

    fn encode_image(&self, path: &Path) -> super::VisionEncoderResult<Tensor> {
        Clip::encode_image(self, path)
    }
}

async fn load_tokenizer(hub: &Hub) -> ClipResult<PathBuf> {
    let mf = ModelFile::clip_tokenizer();
    hub.get_model_file(mf.name, mf.revision, &mf.file)
        .await
        .map_err(ClipError::DownloadTokenizer)
}

async fn load_model_weights(hub: &Hub) -> ClipResult<PathBuf> {
    let mf = ModelFile::clip_model();
    hub.get_model_file(mf.name, mf.revision, &mf.file)
        .await
        .map_err(ClipError::DownloadModel)
}
