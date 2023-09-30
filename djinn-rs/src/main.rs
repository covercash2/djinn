use clap::Parser;
use yolov8::model::{YoloV8, YoloV8Pose};

use crate::yolov8::args::{run, YoloTask};

mod coco_classes;
mod font;
mod yolov8;

fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = yolov8::args::Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    match args.task {
        YoloTask::Detect => run::<YoloV8>(args)?,
        YoloTask::Pose => run::<YoloV8Pose>(args)?,
    }
    Ok(())
}
