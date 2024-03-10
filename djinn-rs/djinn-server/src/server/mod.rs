use axum::{
    routing::{get, post, IntoMakeService},
    Router,
};
use derive_new::new;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, future::IntoFuture, net::SocketAddr, sync::Arc};
use tracing::{instrument, Instrument, Level};

#[derive(new, Debug)]
pub struct Context {
    config: Config,
}

#[derive(new, Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    socker_addr: SocketAddr,
}

#[derive(new)]
pub struct HttpServer {
    context: Arc<Context>,
}

#[instrument]
async fn health_check_handler() -> &'static str {
    tracing::debug!("health checked");
    "OK"
}

#[instrument]
async fn complete() -> &'static str {
    tracing::debug!("complete");
    "nothing"
}

fn build_service() -> IntoMakeService<Router> {
    let router = Router::new()
        .route(
            &ServiceRoutes::HealthCheck.to_string(),
            get(health_check_handler),
        )
        .route(&ServiceRoutes::Complete.to_string(), post(complete));

    router.into_make_service()
}

enum ServiceRoutes {
    HealthCheck,
    Complete,
}

impl Display for ServiceRoutes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceRoutes::HealthCheck => write!(f, "/health-check"),
            ServiceRoutes::Complete => write!(f, "/complete"),
        }
    }
}

impl HttpServer {
    pub async fn start(self) -> anyhow::Result<()> {
        let context = self.context;
        let socket_addr = context.config.socker_addr;

        let listener = tokio::net::TcpListener::bind(&socket_addr).await?;

        let server_span = tracing::span!(Level::INFO, "server span");
        tracing::info!("starting server on {socket_addr}");

        axum::serve(listener, build_service())
            .into_future()
            .instrument(server_span)
            .await?;

        tracing::info!("HTTP server shutdown");

        Ok(())
    }
}
