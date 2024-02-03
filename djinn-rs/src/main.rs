extern crate accelerate_src;

use candle_core::Device;
use clap::{Parser, Subcommand};

mod coco_classes;
mod error;
mod font;
mod llama;
mod mistral;
mod yolov8;
mod token_output_stream;
mod util;

#[derive(Parser)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,
    /// The model architecture used
    #[command(subcommand)]
    architecture: Architecture,
}

#[derive(Subcommand)]
enum Architecture {
    Yolov8(yolov8::args::Args),
    Llama(llama::Args),
    Mistral(mistral::Args),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    match args.architecture {
        Architecture::Yolov8(yolo_args) => yolov8::run(device, yolo_args)?,
        Architecture::Llama(llama_args) => llama::run(device, llama_args).await?,
        Architecture::Mistral(mistral_args) => mistral::run(device, mistral_args).await?,
    }
    Ok(())
}
