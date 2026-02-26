use axum::{
    error_handling::HandleErrorLayer,
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
use tower::{timeout::TimeoutLayer, BoxError, ServiceBuilder, ServiceExt};
use tower_http::{services::ServeDir, trace::TraceLayer};
use tracing::{instrument, Instrument, Level, Span};
use utoipa_swagger_ui::SwaggerUi;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(1);

use crate::clip::ROUTE_CLIP;
use crate::complete::{ROUTE_COMPLETE, ROUTE_COMPLETE_STREAM};
use crate::openapi::ApiDoc;
use crate::ui::ROUTE_UI_COMPLETE;

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

    // Timed routes: health-check, complete (blocking), clip, ui.
    // The 1-second TimeoutLayer lives here so it does NOT affect the SSE stream.
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
        .route(
            &ServiceRoutes::UiComplete.to_string(),
            post(crate::ui::ui_complete),
        )
        .with_state(context.clone())
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(|_: BoxError| async {
                    StatusCode::REQUEST_TIMEOUT
                }))
                .layer(TimeoutLayer::new(REQUEST_TIMEOUT)),
        );

    // Streaming route: no timeout — the connection stays open until generation finishes.
    let stream_router = Router::new()
        .route(
            ROUTE_COMPLETE_STREAM,
            post(crate::complete::stream_complete),
        )
        .with_state(context);

    let router = Router::new()
        .merge(api_router)
        .merge(stream_router)
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
    UiComplete,
}

impl Display for ServiceRoutes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceRoutes::HealthCheck => write!(f, "/health-check"),
            ServiceRoutes::Complete => write!(f, "{}", ROUTE_COMPLETE),
            ServiceRoutes::Clip => write!(f, "{}", ROUTE_CLIP),
            ServiceRoutes::UiComplete => write!(f, "{}", ROUTE_UI_COMPLETE),
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
