use std::path::PathBuf;

use candle_core::{DType, Device, Result, Tensor};
use crate::hf_hub_ext::HubError;

pub mod clip;
pub mod config;
pub mod error;
pub mod gen;
pub mod siglip;
pub mod vision_encoder_ext;

/// Shared error type for all vision-language encoder models (CLIP, SigLIP, …).
///
/// Variants that are specific to a single encoder are documented as such.
#[derive(Debug, thiserror::Error)]
pub enum VisionEncoderError {
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

    /// CLIP-specific: the CLIP tokenizer vocabulary is missing the expected pad token.
    #[error("missing pad token in tokenizer")]
    MissingPadToken,

    #[error("couldn't download model weights: {0}")]
    DownloadModel(HubError),

    #[error("failed to load image from {path}: {source}")]
    LoadImage {
        path: PathBuf,
        source: image::ImageError,
    },

    #[error("couldn't decode image from bytes: {0}")]
    LoadImageBytes(image::ImageError),

    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub type VisionEncoderResult<T> = std::result::Result<T, VisionEncoderError>;

/// Shared interface for vision-language encoder models (CLIP, SigLIP, …).
///
/// Implementors provide text and image encoding methods that return comparable
/// feature vectors, allowing a single downstream similarity computation.
///
/// Errors are reported as [`VisionEncoderResult`] / [`VisionEncoderError`].
pub trait VisionEncoder {
    fn encode_text(&self, text: &str) -> VisionEncoderResult<Tensor>;
    fn encode_image(&self, path: &std::path::Path) -> VisionEncoderResult<Tensor>;
}

/// Saves an image to disk using the image crate, this expects an input with shape
/// (c, height, width).
pub fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        candle_core::bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => candle_core::bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(candle_core::Error::wrap)?;
    Ok(())
}

/// Load a set of images
pub fn load_images<P: AsRef<std::path::Path>>(
    paths: &[P],
    image_size: usize,
) -> anyhow::Result<Tensor> {
    paths.iter()
        .map(|p| load_image(p, image_size))
        .collect::<anyhow::Result<Vec<Tensor>>>()
        .and_then(|images| Tensor::stack(&images, 0).map_err(|e| anyhow::anyhow!(e)))
}

/// Load an image into a [`Tensor`]
pub fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_image_round_trip() {
        // Build a small (3, 4, 4) u8 tensor representing a solid-colour image.
        let pixels: Vec<u8> = (0u8..48).collect();
        let img = Tensor::from_vec(pixels, (3usize, 4usize, 4usize), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap();

        let tmp = tempfile::NamedTempFile::with_suffix(".png").unwrap();
        save_image(&img, tmp.path()).expect("save_image should succeed");

        // load_image normalises to f32 in [-1, 1]; just verify the shape.
        let loaded = load_image(tmp.path(), 4).expect("load_image should succeed");
        let (c, h, w) = loaded.dims3().unwrap();
        assert_eq!((c, h, w), (3, 4, 4));
    }

    #[test]
    fn save_image_rejects_wrong_channel_count() {
        // A (4, 4, 4) tensor has 4 channels – save_image must reject it.
        let pixels: Vec<u8> = vec![0u8; 64];
        let img = Tensor::from_vec(pixels, (4usize, 4usize, 4usize), &Device::Cpu).unwrap();
        let tmp = tempfile::NamedTempFile::with_suffix(".png").unwrap();
        assert!(save_image(&img, tmp.path()).is_err());
    }

    #[test]
    fn load_images_stacks_multiple_images() {
        let pixels: Vec<u8> = (0u8..48).collect();
        let img_tensor = Tensor::from_vec(pixels, (3usize, 4usize, 4usize), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap();

        let tmp1 = tempfile::NamedTempFile::with_suffix(".png").unwrap();
        let tmp2 = tempfile::NamedTempFile::with_suffix(".png").unwrap();
        save_image(&img_tensor, tmp1.path()).unwrap();
        save_image(&img_tensor, tmp2.path()).unwrap();

        let stacked = load_images(&[tmp1.path(), tmp2.path()], 4).unwrap();
        // Batch of 2 images → shape (2, 3, 4, 4).
        assert_eq!(stacked.dims(), &[2, 3, 4, 4]);
    }

    /// Verifies the `VisionEncoder` trait can be used through a mock implementation,
    /// exercising dynamic dispatch and the shared `run_encoder`-style pattern.
    #[test]
    fn vision_encoder_trait_works_with_mock() {
        use candle_core::Device;
        use std::path::Path;

        struct MockEncoder {
            text_vec: Vec<f32>,
            image_vec: Vec<f32>,
        }

        impl VisionEncoder for MockEncoder {
            fn encode_text(&self, _text: &str) -> VisionEncoderResult<Tensor> {
                Ok(Tensor::new(self.text_vec.as_slice(), &Device::Cpu)
                    .expect("test tensor creation should not fail"))
            }

            fn encode_image(&self, _path: &Path) -> VisionEncoderResult<Tensor> {
                Ok(Tensor::new(self.image_vec.as_slice(), &Device::Cpu)
                    .expect("test tensor creation should not fail"))
            }
        }

        let encoder = MockEncoder {
            text_vec: vec![1.0f32, 0.0, 0.0],
            image_vec: vec![1.0f32, 0.0, 0.0],
        };

        let text_feat = encoder.encode_text("test prompt").unwrap();
        let image_feat = encoder.encode_image(Path::new("/fake/path.png")).unwrap();

        // Identical vectors → cosine similarity of 1.0
        let sim = crate::tensor_ext::cosine_similarity(&text_feat, &image_feat).unwrap();
        assert!((sim - 1.0).abs() < 1e-5, "expected similarity ~1.0, got {sim}");
    }
}
