pub use server::{Config, Context, HttpServer};
use std::sync::Arc;

mod server;

pub async fn run_server(config: Config) -> anyhow::Result<()> {
    let context = Context::new(config);
    let server = HttpServer::new(Arc::from(context));

    server.start().await
}
