use clap::Parser;

use djinn_core::yolov8::{Which, YoloTask};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Model weights, in safetensors format.
    #[arg(long)]
    pub model: Option<String>,
    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::S)]
    pub which: Which,
    pub images: Vec<String>,
    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.25)]
    pub confidence_threshold: f32,
    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.45)]
    pub nms_threshold: f32,
    /// The task to be run.
    #[arg(long, default_value = "detect")]
    pub task: YoloTask,
    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 14)]
    pub legend_size: u32,
}
