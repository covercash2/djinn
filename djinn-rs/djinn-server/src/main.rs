use server::{Config, Context, HttpServer};
use std::{net::SocketAddr, sync::Arc};

mod server;

pub async fn run_server(addr: SocketAddr) -> anyhow::Result<()> {
    let context = Context::new(Config::new(addr));
    let server = HttpServer::new(Arc::from(context));

    server.start().await
}

fn main() {
    println!("Hello, world!");
}
