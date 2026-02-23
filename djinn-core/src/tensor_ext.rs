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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let v = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        let sim = cosine_similarity(&v, &v).unwrap();
        assert!((sim - 1.0).abs() < 1e-5, "expected ~1.0, got {sim}");
    }

    #[test]
    fn opposite_vectors_have_similarity_minus_one() {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[-1.0f32, -2.0, -3.0], &Device::Cpu).unwrap();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim + 1.0).abs() < 1e-5, "expected ~-1.0, got {sim}");
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = Tensor::new(&[1.0f32, 0.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[0.0f32, 1.0], &Device::Cpu).unwrap();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-5, "expected ~0.0, got {sim}");
    }

    #[test]
    fn works_with_multidimensional_tensors() {
        // 2-D tensors are flattened before the computation.
        let a = Tensor::new(&[[1.0f32, 0.0], [0.0, 0.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[1.0f32, 0.0], [0.0, 0.0]], &Device::Cpu).unwrap();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-5, "expected ~1.0, got {sim}");
    }
}
