//! Extension helpers for [`super::VisionEncoder`].
//!
//! Provides utility functions that operate on any [`VisionEncoder`] implementor
//! without requiring access to the concrete encoder type.

use super::{VisionEncoder, VisionEncoderResult};
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
    Ok(cosine_similarity(&text_feat, &image_feat)
        .map_err(super::VisionEncoderError::Candle)?)
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
            let sim = cosine_similarity(&text_feat, &image_feat)
                .map_err(super::VisionEncoderError::Candle)?;
            Ok((text, sim))
        })
        .collect::<VisionEncoderResult<_>>()?;
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
}
