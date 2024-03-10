use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post, IntoMakeService},
    Json, Router,
};
use derive_builder::Builder;
use derive_new::new;
use djinn_core::{
    lm::Lm,
    mistral::{model::ModelContext, RunConfig},
};
use futures_util::pin_mut;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    future::{self, IntoFuture},
    net::SocketAddr,
    ops::DerefMut,
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::Mutex;
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
    context: Arc<Mutex<Context>>,
}

pub struct Context {
    pub model: ModelContext,
    pub run_config: RunConfig,
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

#[derive(Serialize, Deserialize, Debug)]
struct CompleteResponse {
    output: String,
}

#[instrument(skip(model_context))]
async fn complete(
    State(model_context): State<Arc<Mutex<Context>>>,
    Json(payload): Json<CompleteRequest>,
) -> impl IntoResponse {
    tracing::debug!("complete");

    let span = tracing::info_span!("complete");

    let mut lock = model_context.lock().instrument(span).await;

    let context: &mut Context = lock.deref_mut();

    let stream = context
        .model
        .run(payload.prompt, context.run_config.clone());

    pin_mut!(stream);

    let mut output = String::new();
    while let Some(value) = stream.next().await {
        if let Ok(string_token) = value {
            output.push_str(&string_token);
            tracing::debug!("{string_token}");
        }
    }
    let response = CompleteResponse { output };

    (StatusCode::OK, Json(response))
}

fn build_service(context: Arc<Mutex<Context>>) -> IntoMakeService<Router> {
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
