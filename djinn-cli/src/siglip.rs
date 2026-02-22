use std::path::PathBuf;

use clap::Parser;
use djinn_core::{
    device::Device,
    image::siglip::{SigLip, SigLipArgs},
    tensor_ext::cosine_similarity,
};

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

pub async fn run(args: Args) -> anyhow::Result<()> {
    let siglip_args = SigLipArgs {
        tokenizer: args.tokenizer.unwrap_or_default(),
        device: args.device.try_into()?,
    };

    tracing::info!("Loading SigLIP model...");
    let siglip = SigLip::new(siglip_args).await?;

    tracing::info!(prompt = %args.prompt, "Encoding text...");
    let text_features = siglip.encode_text(&args.prompt)?;

    tracing::info!(image = ?args.image, "Encoding image...");
    let image_features = siglip.encode_image(&args.image)?;

    let score = cosine_similarity(&text_features, &image_features)?;
    println!("Similarity: {score:.4}");

    Ok(())
}
