use djinn_core::image::clip::{Clip, ClipArgs};

pub use crate::vision_encoder::Args;

pub async fn run(args: Args) -> anyhow::Result<()> {
    let Args { prompt, image, tokenizer, device } = args;
    let clip_args = ClipArgs {
        tokenizer: tokenizer.unwrap_or_default(),
        device: device.try_into()?,
    };

    tracing::info!("Loading CLIP model...");
    let clip = Clip::new(clip_args).await?;

    crate::vision_encoder::run_encoder(&clip, &prompt, &image)
}
