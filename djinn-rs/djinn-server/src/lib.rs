pub use server::{Config, Context, HttpServer};
use tracing::instrument;
use std::sync::Arc;

mod server;

#[instrument]
pub async fn run_server(config: Config) -> anyhow::Result<()> {
    let context = Context::new(config);

    tracing::debug!("starting server with context: {context:?}");

    let server = HttpServer::new(Arc::from(context));

    server.start().await
}
