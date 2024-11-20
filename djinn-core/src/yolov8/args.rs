use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use image::DynamicImage;

use super::{
    model::{Multiples, YoloV8, YoloV8Pose},
    report_detect, report_pose,
};

#[derive(Clone, Copy, ValueEnum, Debug)]
pub enum Which {
    N,
    S,
    M,
    L,
    X,
}

#[derive(Clone, Copy, ValueEnum, Debug)]
pub enum YoloTask {
    Detect,
    Pose,
}

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

impl Args {
    pub fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("lmz/candle-yolo-v8".to_string());
                let size = match self.which {
                    Which::N => "n",
                    Which::S => "s",
                    Which::M => "m",
                    Which::L => "l",
                    Which::X => "x",
                };
                let task = match self.task {
                    YoloTask::Pose => "-pose",
                    YoloTask::Detect => "",
                };
                api.get(&format!("yolov8{size}{task}.safetensors"))?
            }
        };
        Ok(path)
    }
}

pub trait Task: Module + Sized {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self>;
    fn report(
        pred: &Tensor,
        img: DynamicImage,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> Result<DynamicImage>;
}

impl Task for YoloV8 {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self> {
        YoloV8::load(vb, multiples, /* num_classes=*/ 80)
    }

    fn report(
        pred: &Tensor,
        img: DynamicImage,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> Result<DynamicImage> {
        report_detect(
            pred,
            img,
            w,
            h,
            confidence_threshold,
            nms_threshold,
            legend_size,
        )
    }
}

impl Task for YoloV8Pose {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self> {
        YoloV8Pose::load(vb, multiples, /* num_classes=*/ 1, (17, 3))
    }

    fn report(
        pred: &Tensor,
        img: DynamicImage,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        _legend_size: u32,
    ) -> Result<DynamicImage> {
        report_pose(pred, img, w, h, confidence_threshold, nms_threshold)
    }
}
