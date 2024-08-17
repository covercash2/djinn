use axum::{
    extract::{FromRequest, MatchedPath},
    handler::HandlerWithoutStateExt,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post, IntoMakeService},
    Router,
};
use derive_builder::Builder;
use derive_new::new;
use djinn_core::lm::mistral::RunConfig;
use djinn_core::lm::model::ModelContext;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display, future::IntoFuture, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration,
};
use tokio::sync::Mutex;
use tower::ServiceExt;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{instrument, Instrument, Level, Span};

use crate::handlers::ROUTE_COMPLETE;

#[derive(FromRequest)]
#[from_request(via(axum::Json), rejection(crate::error::Error))]
pub struct Json<T>(pub T);

impl<T> IntoResponse for Json<T>
where
    axum::Json<T>: IntoResponse,
{
    fn into_response(self) -> axum::response::Response {
        axum::Json(self.0).into_response()
    }
}

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

async fn not_found() -> (StatusCode, &'static str) {
    (StatusCode::NOT_FOUND, "Not found")
}

fn build_service(context: Arc<Mutex<Context>>) -> IntoMakeService<Router> {
    let router = Router::new()
        .route(
            &ServiceRoutes::HealthCheck.to_string(),
            get(health_check_handler),
        )
        .route(
            &ServiceRoutes::Complete.to_string(),
            post(crate::handlers::complete),
        )
        .fallback_service(
            ServeDir::new("./djinn-server/assets")
                .not_found_service(not_found.into_service())
                .map_request(|request: Request<_>| {
                    tracing::debug!(?request);
                    request
                }),
        )
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request<_>| {
                    let matched_path = request
                        .extensions()
                        .get::<MatchedPath>()
                        .map(MatchedPath::as_str);

                    tracing::debug_span!(
                        "http_request",
                        method = ?request.method(),
                        matched_path,
                    )
                })
                .on_response(|response: &Response, _latency: Duration, span: &Span| {
                    let status: StatusCode = response.status();
                    tracing::debug!(?status);
                    match status {
                        StatusCode::UNSUPPORTED_MEDIA_TYPE => {
                            let content_type = response
                                .headers()
                                .get("content-type")
                                .map(|header| header.to_str().unwrap_or("weird decode error"))
                                .unwrap_or("unknown content-type")
                                .to_string();
                            span.record("media_type", &content_type);
                            tracing::warn!(?status, content_type, "unsupported media type");
                        }
                        _ => {}
                    }
                }),
        )
        .with_state(context);

    router.into_make_service()
}

enum ServiceRoutes {
    HealthCheck,
    Complete,
    CompleteForm,
    Index,
}

impl Display for ServiceRoutes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceRoutes::HealthCheck => write!(f, "/health-check"),
            ServiceRoutes::Complete => write!(f, "{}", ROUTE_COMPLETE),
            ServiceRoutes::CompleteForm => write!(f, "/complete-form"),
            ServiceRoutes::Index => write!(f, "/"),
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
