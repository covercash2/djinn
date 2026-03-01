//! Extension helpers for [`super::VisionEncoder`].
//!
//! Provides utility functions that operate on any [`VisionEncoder`] implementor
//! without requiring access to the concrete encoder type.

use super::{VisionEncoder, VisionEncoderError, VisionEncoderResult};
use crate::tensor_ext::cosine_similarity;

/// Compute the cosine similarity between a text prompt and an image using any
/// [`VisionEncoder`].
///
/// Returns a scalar in `[-1.0, 1.0]`; higher values indicate greater
/// semantic similarity.
pub fn text_image_similarity(
    encoder: &dyn VisionEncoder,
    text: &str,
    image_path: &std::path::Path,
) -> VisionEncoderResult<f32> {
    let text_feat = encoder.encode_text(text)?;
    let image_feat = encoder.encode_image(image_path)?;
    cosine_similarity(&text_feat, &image_feat).map_err(VisionEncoderError::Candle)
}

/// Rank a slice of text prompts against a single image and return them sorted
/// by descending cosine similarity.
pub fn rank_texts_by_image<'a>(
    encoder: &dyn VisionEncoder,
    texts: &[&'a str],
    image_path: &std::path::Path,
) -> VisionEncoderResult<Vec<(&'a str, f32)>> {
    let image_feat = encoder.encode_image(image_path)?;
    let mut scored: Vec<(&str, f32)> = texts
        .iter()
        .map(|&text| {
            let text_feat = encoder.encode_text(text)?;
            let sim =
                cosine_similarity(&text_feat, &image_feat).map_err(VisionEncoderError::Candle)?;
            Ok((text, sim))
        })
        .collect::<VisionEncoderResult<_>>()?;
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use std::path::Path;

    struct MockEncoder {
        text_feat: Vec<f32>,
        image_feat: Vec<f32>,
    }

    impl VisionEncoder for MockEncoder {
        fn encode_text(&self, _: &str) -> VisionEncoderResult<Tensor> {
            Ok(Tensor::new(self.text_feat.as_slice(), &Device::Cpu).unwrap())
        }
        fn encode_image(&self, _: &Path) -> VisionEncoderResult<Tensor> {
            Ok(Tensor::new(self.image_feat.as_slice(), &Device::Cpu).unwrap())
        }
        fn encode_image_from_bytes(&self, _: &[u8]) -> VisionEncoderResult<Tensor> {
            Ok(Tensor::new(self.image_feat.as_slice(), &Device::Cpu).unwrap())
        }
    }

    #[test]
    fn identical_vectors_have_similarity_one() {
        let enc = MockEncoder {
            text_feat: vec![1.0, 0.0, 0.0],
            image_feat: vec![1.0, 0.0, 0.0],
        };
        let sim = text_image_similarity(&enc, "test", Path::new("/fake")).unwrap();
        assert!((sim - 1.0).abs() < 1e-5, "expected ~1.0, got {sim}");
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let enc = MockEncoder {
            text_feat: vec![1.0, 0.0, 0.0],
            image_feat: vec![0.0, 1.0, 0.0],
        };
        let sim = text_image_similarity(&enc, "test", Path::new("/fake")).unwrap();
        assert!(sim.abs() < 1e-5, "expected ~0.0, got {sim}");
    }

    #[test]
    fn rank_texts_sorts_by_descending_similarity() {
        struct SelectiveEncoder;
        impl VisionEncoder for SelectiveEncoder {
            fn encode_text(&self, text: &str) -> VisionEncoderResult<Tensor> {
                let v: &[f32] = if text == "best" {
                    &[1.0, 0.0, 0.0]
                } else {
                    &[0.0, 1.0, 0.0]
                };
                Ok(Tensor::new(v, &Device::Cpu).unwrap())
            }
            fn encode_image(&self, _: &Path) -> VisionEncoderResult<Tensor> {
                Ok(Tensor::new(&[1.0f32, 0.0, 0.0], &Device::Cpu).unwrap())
            }
            fn encode_image_from_bytes(&self, _: &[u8]) -> VisionEncoderResult<Tensor> {
                Ok(Tensor::new(&[1.0f32, 0.0, 0.0], &Device::Cpu).unwrap())
            }
        }
        let ranked =
            rank_texts_by_image(&SelectiveEncoder, &["worst", "best"], Path::new("/fake")).unwrap();
        assert_eq!(ranked[0].0, "best");
        assert_eq!(ranked[1].0, "worst");
        assert!(ranked[0].1 > ranked[1].1);
    }

    #[test]
    fn rank_texts_empty_slice_returns_empty_vec() {
        let enc = MockEncoder {
            text_feat: vec![],
            image_feat: vec![1.0, 0.0, 0.0],
        };
        let ranked = rank_texts_by_image(&enc, &[], Path::new("/fake")).unwrap();
        assert!(ranked.is_empty());
    }
}
