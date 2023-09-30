use clap::{Parser, Subcommand};

mod coco_classes;
mod font;
mod yolov8;

#[derive(Parser)]
struct Args {
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
}

fn main() -> anyhow::Result<()> {
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

    match args.architecture {
        Architecture::Yolov8(yolo_args) => yolov8::run(yolo_args)?,
    }
    Ok(())
}
