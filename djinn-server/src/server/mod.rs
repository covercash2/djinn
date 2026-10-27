use axum::{
    error_handling::HandleErrorLayer,
    extract::{FromRequest, MatchedPath},
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use derive_builder::Builder;
use derive_new::new;
use djinn_models::lm::LanguageModel;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display, future::IntoFuture, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration,
};
use tokio::sync::Mutex;
use tower::{timeout::TimeoutLayer, BoxError, ServiceBuilder};
use tower_http::trace::TraceLayer;
use tracing::{instrument, Instrument, Level, Span};
use utoipa_swagger_ui::SwaggerUi;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(1);

use crate::chat::{ChatState, ROUTE_CHAT, ROUTE_MODELS};
use crate::complete::{ROUTE_COMPLETE, ROUTE_COMPLETE_STREAM};
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
    pub socket_addr: SocketAddr,
    pub model_config: PathBuf,
}

#[derive(Builder)]
pub struct HttpServer {
    #[builder(setter(into))]
    config: Arc<Config>,
    state: AppState,
}

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<Mutex<Box<dyn LanguageModel>>>,
    /// Identifier of the loaded model, cached so listing models never waits for a generation.
    pub model_id: Arc<str>,
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

pub(crate) fn build_router(state: AppState) -> Router {
    use utoipa::OpenApi;

    let chat_state = ChatState {
        model: state.model.clone(),
        model_id: state.model_id.clone(),
    };

    // Fast routes: neither the health check nor the model list runs inference, so a 1s timeout is
    // appropriate.
    let fast_router = Router::new()
        .route(
            &ServiceRoutes::HealthCheck.to_string(),
            get(health_check_handler),
        )
        .route(ROUTE_MODELS, get(crate::chat::list_models))
        .with_state(chat_state.clone())
        .layer(
            ServiceBuilder::new()
                .layer(HandleErrorLayer::new(|_: BoxError| async {
                    StatusCode::REQUEST_TIMEOUT
                }))
                .layer(TimeoutLayer::new(REQUEST_TIMEOUT)),
        );

    // Inference routes: no timeout — generation time is unbounded.
    // Model gets Arc<Mutex<Box<dyn LanguageModel>>> — mutex required since run() takes &mut self.
    let inference_router = Router::new()
        .route(
            &ServiceRoutes::Complete.to_string(),
            post(crate::complete::complete),
        )
        .route(
            ROUTE_COMPLETE_STREAM,
            post(crate::complete::stream_complete),
        )
        .with_state(state.model);

    // Chat is an inference route too: no timeout, same single-model lock.
    let chat_router = Router::new()
        .route(ROUTE_CHAT, post(crate::chat::chat_completions))
        .with_state(chat_state);

    let router = Router::new()
        .merge(fast_router)
        .merge(inference_router)
        .merge(chat_router)
        .merge(SwaggerUi::new("/swagger-ui").url("/api-doc/openapi.json", ApiDoc::openapi()))
        .fallback(not_found)
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

    router
}

enum ServiceRoutes {
    HealthCheck,
    Complete,
}

impl Display for ServiceRoutes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceRoutes::HealthCheck => write!(f, "/health-check"),
            ServiceRoutes::Complete => write!(f, "{}", ROUTE_COMPLETE),
        }
    }
}

impl HttpServer {
    pub async fn start(self) -> anyhow::Result<()> {
        let socket_addr = self.config.socket_addr;

        let listener = tokio::net::TcpListener::bind(&socket_addr).await?;

        let server_span = tracing::span!(Level::INFO, "server span");
        tracing::info!("starting server on {socket_addr}");

        axum::serve(listener, build_router(self.state).into_make_service())
            .into_future()
            .instrument(server_span)
            .await?;

        tracing::info!("HTTP server shutdown");

        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use axum::http::{self, Request};
    use djinn_core::lm::config::RunConfig;
    use futures::stream;
    use std::pin::Pin;
    use tokio_stream::Stream;
    use tower::ServiceExt;

    pub(crate) struct MockLanguageModel {
        pub tokens: Vec<String>,
    }

    impl LanguageModel for MockLanguageModel {
        fn run(
            &mut self,
            _prompt: String,
            _config: RunConfig,
        ) -> Pin<Box<dyn Stream<Item = Result<String, djinn_models::Error>> + Send + '_>> {
            let tokens = self.tokens.clone();
            Box::pin(stream::iter(tokens.into_iter().map(Ok)))
        }
    }

    pub(crate) fn test_state(tokens: Vec<&str>) -> AppState {
        AppState {
            model: Arc::new(Mutex::new(Box::new(MockLanguageModel {
                tokens: tokens.into_iter().map(str::to_owned).collect(),
            }))),
            model_id: "test-model".into(),
        }
    }

    pub(crate) fn test_app(tokens: Vec<&str>) -> Router {
        build_router(test_state(tokens))
    }

    struct ErrorLanguageModel;

    impl LanguageModel for ErrorLanguageModel {
        fn run(
            &mut self,
            _prompt: String,
            _config: RunConfig,
        ) -> Pin<Box<dyn Stream<Item = Result<String, djinn_models::Error>> + Send + '_>> {
            Box::pin(stream::once(async {
                Err(djinn_models::Error::Anyhow(anyhow::anyhow!("model error")))
            }))
        }
    }

    fn error_test_app() -> Router {
        build_router(AppState {
            model: Arc::new(Mutex::new(Box::new(ErrorLanguageModel))),
            model_id: "test-model".into(),
        })
    }

    #[tokio::test]
    async fn health_check_returns_200() {
        let app = test_app(vec![]);
        let request = Request::builder()
            .uri("/health-check")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);
    }

    #[tokio::test]
    async fn complete_returns_json_with_prompt_and_output() {
        let app = test_app(vec!["hello", " world"]);
        let body = serde_json::json!({ "prompt": "say hi" }).to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/complete")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);

        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["prompt"], "say hi");
        assert_eq!(json["output"], "hello world");
    }

    #[tokio::test]
    async fn complete_empty_prompt_still_returns_200() {
        let app = test_app(vec!["response"]);
        let body = serde_json::json!({ "prompt": "" }).to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/complete")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);
    }

    #[tokio::test]
    async fn complete_missing_content_type_returns_415() {
        let app = test_app(vec![]);
        let request = Request::builder()
            .method("POST")
            .uri("/complete")
            .body(axum::body::Body::from(r#"{"prompt":"hi"}"#))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[tokio::test]
    async fn stream_complete_returns_sse_content_type() {
        let app = test_app(vec!["tok1", "tok2"]);
        let body = serde_json::json!({ "prompt": "hi" }).to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/complete/stream")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);
        let ct = response.headers().get(http::header::CONTENT_TYPE).unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }

    #[tokio::test]
    async fn stream_complete_body_contains_tokens_and_done() {
        let app = test_app(vec!["hello", " world"]);
        let body = serde_json::json!({ "prompt": "hi" }).to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/complete/stream")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let body_str = std::str::from_utf8(&bytes).unwrap();
        assert!(
            body_str.contains("hello"),
            "missing first token: {body_str}"
        );
        assert!(
            body_str.contains(" world"),
            "missing second token: {body_str}"
        );
        assert!(
            body_str.contains("event: done"),
            "missing done event: {body_str}"
        );
    }

    #[tokio::test]
    async fn clip_route_is_removed() {
        let app = test_app(vec![]);
        let request = Request::builder()
            .method("POST")
            .uri("/clip")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from("{}"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn root_path_is_not_found() {
        let app = test_app(vec![]);
        let request = Request::builder()
            .uri("/")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        assert_eq!(&bytes[..], b"Not found");
    }

    #[tokio::test]
    async fn complete_model_error_returns_500() {
        let app = error_test_app();
        let body = serde_json::json!({ "prompt": "hi" }).to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/complete")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn stream_complete_model_error_sends_error_event() {
        let app = error_test_app();
        let body = serde_json::json!({ "prompt": "hi" }).to_string();
        let request = Request::builder()
            .method("POST")
            .uri("/complete/stream")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        let bytes = http_body_util::BodyExt::collect(response.into_body())
            .await
            .unwrap()
            .to_bytes();
        let body_str = std::str::from_utf8(&bytes).unwrap();
        assert!(
            body_str.contains("event: error"),
            "missing error event: {body_str}"
        );
    }
}
