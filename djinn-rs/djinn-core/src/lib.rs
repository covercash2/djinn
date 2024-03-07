extern crate accelerate_src;

use std::{net::{SocketAddr, SocketAddrV4}, sync::Arc};

use clap::{Parser, Subcommand};
use server::{Config, Context, HttpServer};

mod coco_classes;
pub mod device;
mod error;
mod font;
pub mod llama;
pub mod lm;
pub mod mistral;
mod server;
mod text_generator;
mod token_output_stream;
mod util;
pub mod yolov8;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    runner: Runner,
}

#[derive(Subcommand)]
enum Runner {
    Server,
    SingleRun(Args),
}

#[derive(Parser)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[deprecated(note = "this should be removed and moved to the specific model config")]
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

// TODO re-enable server
pub async fn run_server() -> anyhow::Result<()> {
    let addr: SocketAddrV4 = "127.0.0.1:8090".parse()?;
    let context = Context::new(Config::new(SocketAddr::from(addr)));
    let server = HttpServer::new(Arc::from(context));

    server.start().await
}
