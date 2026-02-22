use djinn_core::image::siglip::{SigLip, SigLipArgs};

pub use crate::vision_encoder::Args;

pub async fn run(args: Args) -> anyhow::Result<()> {
    let Args { prompt, image, tokenizer, device } = args;
    let siglip_args = SigLipArgs {
        tokenizer: tokenizer.unwrap_or_default(),
        device: device.try_into()?,
    };

    tracing::info!("Loading SigLIP model...");
    let siglip = SigLip::new(siglip_args).await?;

    crate::vision_encoder::run_encoder(&siglip, &prompt, &image)
}
