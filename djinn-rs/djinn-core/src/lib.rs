extern crate accelerate_src;

use std::{
    net::{SocketAddr, SocketAddrV4},
    sync::Arc,
};

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

pub async fn run_server() -> anyhow::Result<()> {
    let addr: SocketAddrV4 = "127.0.0.1:8090".parse()?;
    let context = Context::new(Config::new(SocketAddr::from(addr)));
    let server = HttpServer::new(Arc::from(context));

    server.start().await
}
