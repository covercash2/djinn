use axum::{
    Router,
    error_handling::HandleErrorLayer,
    extract::{FromRequest, MatchedPath},
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use derive_builder::Builder;
use djinn_core::values::ModelId;
use djinn_models::lm::LanguageModel;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display, future::IntoFuture, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration,
};
use tokio::sync::Mutex;
use tower::{BoxError, ServiceBuilder, timeout::TimeoutLayer};
use tower_http::trace::TraceLayer;
use tracing::{Instrument, Level, Span, instrument};
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

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Config {
    pub socket_addr: SocketAddr,
    pub model_config: PathBuf,
}

impl Config {
    pub fn new(socket_addr: SocketAddr, model_config: PathBuf) -> Self {
        Self {
            socket_addr,
            model_config,
        }
    }
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
    pub model_id: ModelId,
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

    Router::new()
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
        )
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
    use djinn_core::values::Prompt;
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
            _prompt: Prompt,
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
            model_id: ModelId::from_static("test-model"),
        }
    }

    pub(crate) fn test_app(tokens: Vec<&str>) -> Router {
        build_router(test_state(tokens))
    }

    struct ErrorLanguageModel;

    impl LanguageModel for ErrorLanguageModel {
        fn run(
            &mut self,
            _prompt: Prompt,
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
            model_id: ModelId::from_static("test-model"),
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

    /// Posts `body` to `/complete` and returns the status and the response text.
    async fn post_complete(body: serde_json::Value) -> (http::StatusCode, String) {
        let app = test_app(vec!["ok"]);
        let request = Request::builder()
            .method("POST")
            .uri("/complete")
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn complete_rejects_an_out_of_range_temperature_naming_it() {
        let (status, body) =
            post_complete(serde_json::json!({"prompt": "hi", "temperature": 3})).await;
        assert!(status.is_client_error(), "{status}: {body}");
        assert!(body.contains("temperature"), "{body}");
    }

    #[tokio::test]
    async fn complete_rejects_a_bad_top_p_and_a_zero_sample_length() {
        let (status, body) = post_complete(serde_json::json!({"prompt": "hi", "top_p": 0})).await;
        assert!(
            status.is_client_error() && body.contains("top_p"),
            "{status}: {body}"
        );
        let (status, body) =
            post_complete(serde_json::json!({"prompt": "hi", "sample_len": 0})).await;
        assert!(
            status.is_client_error() && body.contains("max_tokens"),
            "{status}: {body}"
        );
    }

    #[tokio::test]
    async fn complete_with_valid_parameters_still_works() {
        let (status, _) = post_complete(serde_json::json!({
            "prompt": "hi", "temperature": 0.5, "top_p": 0.9, "sample_len": 10
        }))
        .await;
        assert_eq!(status, http::StatusCode::OK);
    }

    #[test]
    fn openapi_publishes_the_sampling_bounds() {
        use utoipa::OpenApi;
        fn find<'a>(value: &'a serde_json::Value, title: &str) -> Option<&'a serde_json::Value> {
            match value {
                serde_json::Value::Object(map) => {
                    if map.get("title").and_then(|t| t.as_str()) == Some(title) {
                        return Some(value);
                    }
                    map.values().find_map(|v| find(v, title))
                }
                serde_json::Value::Array(items) => items.iter().find_map(|v| find(v, title)),
                _ => None,
            }
        }
        let doc = serde_json::to_value(crate::openapi::ApiDoc::openapi()).unwrap();
        let temperature = find(&doc, "temperature").expect("temperature is documented");
        assert_eq!(temperature["minimum"], 0);
        assert_eq!(temperature["maximum"], 2);
        let tokens = find(&doc, "max_tokens").expect("max_tokens is documented");
        assert_eq!(tokens["minimum"], 1);
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
