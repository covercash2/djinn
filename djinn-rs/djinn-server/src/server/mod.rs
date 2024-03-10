use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post, IntoMakeService},
    Json, Router,
};
use derive_builder::Builder;
use derive_new::new;
use djinn_core::mistral::model::ModelContext;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, future::IntoFuture, net::SocketAddr, path::PathBuf, sync::Arc};
use tracing::{instrument, Instrument, Level};

#[derive(new, Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub socker_addr: SocketAddr,
    pub model_config: PathBuf,
}

#[derive(Builder)]
pub struct HttpServer {
    #[builder(setter(into))]
    config: Arc<Config>,
    #[builder(setter(into))]
    context: Arc<ModelContext>,
}

#[instrument]
async fn health_check_handler() -> &'static str {
    tracing::debug!("health checked");
    "OK"
}

#[derive(Serialize, Deserialize, Debug)]
struct CompleteRequest {
    prompt: String,
}

#[instrument]
async fn complete(Json(payload): Json<CompleteRequest>) -> impl IntoResponse {
    tracing::debug!("complete");
    (StatusCode::OK, Json(payload))
}

fn build_service(context: Arc<ModelContext>) -> IntoMakeService<Router> {
    let router = Router::new()
        .route(
            &ServiceRoutes::HealthCheck.to_string(),
            get(health_check_handler),
        )
        .route(&ServiceRoutes::Complete.to_string(), post(complete))
        .with_state(context);

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
        let socket_addr = self.config.socker_addr;

        let listener = tokio::net::TcpListener::bind(&socket_addr).await?;

        let server_span = tracing::span!(Level::INFO, "server span");
        tracing::info!("starting server on {socket_addr}");

        axum::serve(listener, build_service(context))
            .into_future()
            .instrument(server_span)
            .await?;

        tracing::info!("HTTP server shutdown");

        Ok(())
    }
}
