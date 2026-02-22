use candle_core::{Result, Tensor};

/// Computes the cosine similarity between two tensors.
///
/// Both tensors are flattened before the computation, so this works for any
/// shape as long as the total element counts match.  Returns a scalar in
/// `[-1.0, 1.0]`.
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.flatten_all()?;
    let b = b.flatten_all()?;
    let dot = (&a * &b)?.sum_all()?;
    let norm_a = a.sqr()?.sum_all()?.sqrt()?;
    let norm_b = b.sqr()?.sum_all()?.sqrt()?;
    (dot / (norm_a * norm_b)?)?.to_scalar::<f32>()
}
