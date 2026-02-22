use std::path::Path;
use std::path::PathBuf;

use clap::Parser;
use djinn_core::{
    device::Device,
    image::VisionEncoder,
    tensor_ext::cosine_similarity,
};

/// Shared CLI arguments for vision-language similarity models (CLIP, SigLIP, …).
#[derive(Parser)]
pub struct Args {
    /// Text prompt to compare against the image.
    #[arg(long)]
    pub prompt: String,

    /// Path to the image file to compare against the prompt.
    #[arg(long)]
    pub image: PathBuf,

    /// Path to a local tokenizer file (downloads from HF Hub if not provided).
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Compute device to run inference on.
    #[arg(long, default_value = "cpu")]
    pub device: Device,
}

/// Encode `prompt` and `image` with `encoder`, compute cosine similarity, and print it.
pub fn run_encoder(encoder: &impl VisionEncoder, prompt: &str, image: &Path) -> anyhow::Result<()> {
    tracing::info!(prompt = %prompt, "Encoding text...");
    let text_features = encoder.encode_text(prompt)?;

    tracing::info!(image = ?image, "Encoding image...");
    let image_features = encoder.encode_image(image)?;

    let score = cosine_similarity(&text_features, &image_features)?;
    println!("Similarity: {score:.4}");

    Ok(())
}
