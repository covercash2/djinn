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
use djinn_core::image::clip::Clip;
use djinn_core::lm::model::ModelContext;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display, future::IntoFuture, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration,
};
use tokio::sync::Mutex;
use tower::ServiceExt;
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{instrument, Instrument, Level, Span};
use utoipa_swagger_ui::SwaggerUi;

use crate::clip::ROUTE_CLIP;
use crate::complete::ROUTE_COMPLETE;
use crate::openapi::ApiDoc;

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

#[derive(new, Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
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
    pub clip: Clip,
}

/// Returns `200 OK` when the server is running.
#[utoipa::path(
    get,
    path = "/health-check",
    responses(
        (status = 200, description = "Server is running", body = str),
    ),
    tag = "health",
)]
#[instrument]
pub(crate) async fn health_check_handler() -> &'static str {
    tracing::debug!("health checked");
    "OK"
}

async fn not_found() -> (StatusCode, &'static str) {
    (StatusCode::NOT_FOUND, "Not found")
}

fn build_service(context: Arc<Mutex<Context>>) -> IntoMakeService<Router> {
    use utoipa::OpenApi;

    // API routes require shared state; resolve state first so the router
    // becomes Router<()>, which can then be merged with stateless routers.
    let api_router = Router::new()
        .route(
            &ServiceRoutes::HealthCheck.to_string(),
            get(health_check_handler),
        )
        .route(
            &ServiceRoutes::Complete.to_string(),
            post(crate::complete::complete),
        )
        .route(
            &ServiceRoutes::Clip.to_string(),
            post(crate::clip::clip_similarity),
        )
        .with_state(context);

    let router = Router::new()
        .merge(api_router)
        .merge(
            SwaggerUi::new("/swagger-ui")
                .url("/api-doc/openapi.json", ApiDoc::openapi()),
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
                    if let StatusCode::UNSUPPORTED_MEDIA_TYPE = status {
                        let content_type = response
                            .headers()
                            .get("content-type")
                            .map(|header| header.to_str().unwrap_or("weird decode error"))
                            .unwrap_or("unknown content-type")
                            .to_string();
                        span.record("media_type", &content_type);
                        tracing::warn!(?status, content_type, "unsupported media type");
                    }
                }),
        );

    router.into_make_service()
}

enum ServiceRoutes {
    HealthCheck,
    Complete,
    Clip,
}

impl Display for ServiceRoutes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceRoutes::HealthCheck => write!(f, "/health-check"),
            ServiceRoutes::Complete => write!(f, "{}", ROUTE_COMPLETE),
            ServiceRoutes::Clip => write!(f, "{}", ROUTE_CLIP),
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
