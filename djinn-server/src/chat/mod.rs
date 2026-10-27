//! OpenAI-compatible chat endpoints: `POST /v1/chat/completions` and `GET /v1/models`.

use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use djinn_core::chat::{
    validate_request, ChatChunk, ChatMessage, ChatRequest, ChatResponse, Choice, ChunkChoice,
    Delta, ErrorBody, FinishReason, FunctionCallDelta, ModelInfo, ModelList, Role, StopSpec,
    ToolCall, ToolCallDelta, Usage, OBJECT_CHAT_COMPLETION, OBJECT_CHAT_COMPLETION_CHUNK,
};
use djinn_models::lm::chat::{ChatEvent, ChatInput, ChatOutcome, ChatParams};
use djinn_models::lm::LanguageModel;
use futures::{pin_mut, StreamExt};
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tracing::instrument;

pub const ROUTE_CHAT: &str = "/v1/chat/completions";
pub const ROUTE_MODELS: &str = "/v1/models";

/// State shared by the chat routes: the one loaded model, and its id (cached so listing models
/// never has to wait for a running generation).
#[derive(Clone)]
pub struct ChatState {
    pub model: Arc<Mutex<Box<dyn LanguageModel>>>,
    pub model_id: Arc<str>,
}

/// Why a chat request failed. Rendered as an OpenAI-style `{"error": {...}}` body.
#[derive(Debug)]
pub enum ChatError {
    /// The request is malformed or breaks a protocol rule.
    InvalidRequest { message: String, status: StatusCode },
    /// The conversation does not fit the model's context window.
    ContextLengthExceeded(String),
    /// The loaded model is not instruction-tuned.
    ChatUnsupported,
    /// Anything else; details are logged, not returned.
    Internal(djinn_models::Error),
}

impl ChatError {
    fn invalid(message: impl Into<String>) -> Self {
        ChatError::InvalidRequest {
            message: message.into(),
            status: StatusCode::BAD_REQUEST,
        }
    }

    fn body(&self) -> (StatusCode, ErrorBody) {
        match self {
            ChatError::InvalidRequest { message, status } => (
                *status,
                ErrorBody::new("invalid_request_error", message, None),
            ),
            ChatError::ContextLengthExceeded(message) => (
                StatusCode::BAD_REQUEST,
                ErrorBody::new(
                    "invalid_request_error",
                    message,
                    Some("context_length_exceeded"),
                ),
            ),
            ChatError::ChatUnsupported => (
                StatusCode::NOT_IMPLEMENTED,
                ErrorBody::new(
                    "server_error",
                    "the loaded model does not support chat; configure an instruct model",
                    Some("chat_not_supported"),
                ),
            ),
            ChatError::Internal(error) => {
                tracing::error!(%error, "chat generation failed");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ErrorBody::new("server_error", "Something went wrong D:", None),
                )
            }
        }
    }
}

impl From<djinn_models::Error> for ChatError {
    fn from(error: djinn_models::Error) -> Self {
        match error {
            djinn_models::Error::ContextLengthExceeded { .. } => {
                ChatError::ContextLengthExceeded(error.to_string())
            }
            djinn_models::Error::ChatUnsupported => ChatError::ChatUnsupported,
            other => ChatError::Internal(other),
        }
    }
}

impl IntoResponse for ChatError {
    fn into_response(self) -> Response {
        let (status, body) = self.body();
        (status, axum::Json(body)).into_response()
    }
}

impl From<JsonRejection> for ChatError {
    fn from(rejection: JsonRejection) -> Self {
        let status = match &rejection {
            // A body that parses as JSON but does not fit the schema (unknown role, wrong type)
            // is a bad request for this API, not an "unprocessable entity".
            JsonRejection::JsonDataError(_) | JsonRejection::JsonSyntaxError(_) => {
                StatusCode::BAD_REQUEST
            }
            other => other.status(),
        };
        ChatError::InvalidRequest {
            message: rejection.body_text(),
            status,
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn completion_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!("chatcmpl-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

fn to_input(request: ChatRequest) -> ChatInput {
    ChatInput {
        messages: request.messages,
        tools: request.tools.unwrap_or_default(),
        params: ChatParams {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
            stop: request.stop.map(StopSpec::into_vec).unwrap_or_default(),
        },
    }
}

fn response_message(outcome: &ChatOutcome) -> ChatMessage {
    if outcome.tool_calls.is_empty() {
        ChatMessage {
            role: Role::Assistant,
            content: Some(outcome.content.clone().unwrap_or_default()),
            tool_calls: None,
            tool_call_id: None,
        }
    } else {
        ChatMessage::assistant_tool_calls(outcome.content.clone(), outcome.tool_calls.clone())
    }
}

fn tool_call_deltas(calls: &[ToolCall]) -> Vec<ToolCallDelta> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| ToolCallDelta {
            index: Some(index as u32),
            id: Some(call.id.clone()),
            kind: Some(call.kind),
            function: Some(FunctionCallDelta {
                name: Some(call.function.name.clone()),
                arguments: Some(call.function.arguments.clone()),
            }),
        })
        .collect()
}

struct ChunkMaker {
    id: String,
    created: u64,
    model: String,
}

impl ChunkMaker {
    fn chunk(
        &self,
        delta: Delta,
        finish_reason: Option<FinishReason>,
        usage: Option<Usage>,
    ) -> ChatChunk {
        ChatChunk {
            id: self.id.clone(),
            object: OBJECT_CHAT_COMPLETION_CHUNK.to_owned(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage,
        }
    }

    fn event(&self, chunk: &ChatChunk) -> Event {
        Event::default().data(serde_json::to_string(chunk).expect("chunk serializes"))
    }
}

fn error_event(error: &ChatError) -> Event {
    let (_, body) = error.body();
    Event::default().data(serde_json::to_string(&body).expect("error body serializes"))
}

/// Creates a chat completion, as JSON or as a Server-Sent-Events stream.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    request_body = ChatRequest,
    responses(
        (status = 200, description = "Chat completion (JSON, or text/event-stream when `stream` is true)", body = ChatResponse),
        (status = 400, description = "Invalid request or context length exceeded", body = ErrorBody),
        (status = 500, description = "Inference error", body = ErrorBody),
        (status = 501, description = "The loaded model does not support chat", body = ErrorBody),
    ),
    tag = "chat",
)]
#[instrument(skip(state, request))]
pub async fn chat_completions(
    State(state): State<ChatState>,
    request: Result<axum::Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ChatError> {
    let axum::Json(request) = request?;
    validate_request(&request).map_err(|e| ChatError::invalid(e.to_string()))?;

    let streaming = request.is_stream();
    let input = to_input(request);

    if streaming {
        stream_chat(state, input).await
    } else {
        let response = complete_chat(&state, input).await?;
        Ok(axum::Json(response).into_response())
    }
}

async fn complete_chat(state: &ChatState, input: ChatInput) -> Result<ChatResponse, ChatError> {
    let mut model = state.model.lock().await;
    let stream = model.chat(input);
    pin_mut!(stream);

    let mut outcome = None;
    while let Some(event) = stream.next().await {
        if let ChatEvent::Done(done) = event? {
            outcome = Some(done);
        }
    }
    let outcome = outcome.ok_or_else(|| {
        ChatError::Internal(djinn_models::Error::Anyhow(anyhow::anyhow!(
            "model ended without a result"
        )))
    })?;

    Ok(ChatResponse {
        id: completion_id(),
        object: OBJECT_CHAT_COMPLETION.to_owned(),
        created: now(),
        model: state.model_id.to_string(),
        choices: vec![Choice {
            index: 0,
            message: response_message(&outcome),
            finish_reason: outcome.finish,
        }],
        usage: outcome.usage,
    })
}

/// Starts a streaming completion.
///
/// The model is locked and polled by a background task for the whole generation. The first item is
/// awaited *before* the HTTP response is built, so errors that are known up front (such as a
/// conversation that is too long) become proper HTTP error statuses instead of a 200 whose stream
/// contains an error.
async fn stream_chat(state: ChatState, input: ChatInput) -> Result<Response, ChatError> {
    let (events_tx, events_rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), ChatError>>();
    let maker = ChunkMaker {
        id: completion_id(),
        created: now(),
        model: state.model_id.to_string(),
    };

    tokio::spawn(async move {
        let mut model = state.model.lock().await;
        let stream = model.chat(input);
        pin_mut!(stream);

        let mut ready_tx = Some(ready_tx);

        // The opening chunk announces the role, as OpenAI clients expect.
        let opening = maker.event(&maker.chunk(
            Delta {
                role: Some(Role::Assistant),
                content: Some(String::new()),
                tool_calls: None,
            },
            None,
            None,
        ));

        while let Some(item) = stream.next().await {
            // The first item settles whether the request can be served at all.
            let item = match ready_tx.take() {
                None => item,
                Some(ready) => match item {
                    Err(error) => {
                        let _ = ready.send(Err(ChatError::from(error)));
                        return;
                    }
                    Ok(event) => {
                        let _ = ready.send(Ok(()));
                        if events_tx.send(Ok(opening.clone())).await.is_err() {
                            return; // client disconnected
                        }
                        Ok(event)
                    }
                },
            };
            if !send_item(&events_tx, &maker, item).await {
                return;
            }
        }
        // The model ended without a result; make sure the waiting handler is released.
        if let Some(ready) = ready_tx.take() {
            let _ = ready.send(Err(ChatError::Internal(djinn_models::Error::Anyhow(
                anyhow::anyhow!("model produced no output"),
            ))));
        }
    });

    match ready_rx.await {
        Ok(Ok(())) => Ok(Sse::new(ReceiverStream::new(events_rx)).into_response()),
        Ok(Err(error)) => Err(error),
        Err(_) => Err(ChatError::Internal(djinn_models::Error::Anyhow(
            anyhow::anyhow!("generation task ended unexpectedly"),
        ))),
    }
}

/// Sends one model item to the client. Returns `false` when generation should stop (the client is
/// gone, the reply is complete, or an error ended the stream).
async fn send_item(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    maker: &ChunkMaker,
    item: djinn_models::Result<ChatEvent>,
) -> bool {
    match item {
        Ok(ChatEvent::Text(text)) => {
            let chunk = maker.chunk(
                Delta {
                    content: Some(text),
                    ..Delta::default()
                },
                None,
                None,
            );
            tx.send(Ok(maker.event(&chunk))).await.is_ok()
        }
        Ok(ChatEvent::Done(outcome)) => {
            let delta = Delta {
                tool_calls: (!outcome.tool_calls.is_empty())
                    .then(|| tool_call_deltas(&outcome.tool_calls)),
                ..Delta::default()
            };
            let last = maker.chunk(delta, Some(outcome.finish), Some(outcome.usage));
            if tx.send(Ok(maker.event(&last))).await.is_err() {
                return false;
            }
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            false
        }
        Err(error) => {
            let error = ChatError::from(error);
            tracing::error!(?error, "chat stream failed");
            let _ = tx.send(Ok(error_event(&error))).await;
            false
        }
    }
}

/// Lists the loaded model.
#[utoipa::path(
    get,
    path = "/v1/models",
    responses((status = 200, description = "The loaded model", body = ModelList)),
    tag = "chat",
)]
pub async fn list_models(State(state): State<ChatState>) -> axum::Json<ModelList> {
    axum::Json(ModelList {
        object: "list".to_owned(),
        data: vec![ModelInfo {
            id: state.model_id.to_string(),
            object: "model".to_owned(),
            owned_by: "djinn".to_owned(),
        }],
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::{header, Request};
    use axum::Router;
    use djinn_models::lm::chat::ChatStream;
    use djinn_models::testing::ScriptedModel;
    use futures::stream;
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use super::*;
    use crate::server::{build_router, AppState};

    fn app_with(model: impl LanguageModel + 'static) -> Router {
        build_router(AppState {
            model: Arc::new(Mutex::new(Box::new(model))),
            model_id: "test-model".into(),
        })
    }

    async fn send(app: &Router, request: Request<Body>) -> (StatusCode, String) {
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    fn post(body: &Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(ROUTE_CHAT)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn chat(app: &Router, body: Value) -> (StatusCode, Value) {
        let (status, text) = send(app, post(&body)).await;
        (status, serde_json::from_str(&text).unwrap_or(Value::Null))
    }

    fn user(text: &str) -> Value {
        json!({"role": "user", "content": text})
    }

    /// `data:` payloads of an SSE body, in order.
    fn sse_data(body: &str) -> Vec<String> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|data| data.trim_start().to_owned())
            .collect()
    }

    fn chunks(body: &str) -> Vec<Value> {
        sse_data(body)
            .iter()
            .filter(|d| *d != "[DONE]")
            .map(|d| serde_json::from_str(d).unwrap())
            .collect()
    }

    fn content_deltas(chunks: &[Value]) -> Vec<String> {
        chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    }

    fn tool_outcome(content: Option<&str>) -> ChatOutcome {
        ChatOutcome {
            content: content.map(str::to_owned),
            tool_calls: vec![ToolCall::function(
                "call_7",
                "calculator",
                r#"{"expression":"2+2"}"#,
            )],
            finish: FinishReason::ToolCalls,
            usage: Usage::new(11, 4),
        }
    }

    // ---- request handling ------------------------------------------------------------------

    #[tokio::test]
    async fn minimal_request_gets_a_chat_completion() {
        let app = app_with(ScriptedModel::replying("Hello!"));
        let (status, body) = chat(&app, json!({"messages": [user("hi")]})).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "test-model");
        assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert!(body["choices"][0]["message"].get("tool_calls").is_none());
        assert_eq!(body["usage"]["prompt_tokens"], 5);
        assert_eq!(body["usage"]["completion_tokens"], 3);
        assert_eq!(body["usage"]["total_tokens"], 8);
    }

    #[tokio::test]
    async fn empty_messages_are_rejected() {
        let app = app_with(ScriptedModel::replying("x"));
        let (status, body) = chat(&app, json!({"messages": []})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("messages"));
    }

    #[tokio::test]
    async fn unknown_role_is_rejected_with_400() {
        let app = app_with(ScriptedModel::replying("x"));
        let (status, body) = chat(
            &app,
            json!({"messages": [{"role": "robot", "content": "hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn malformed_json_is_a_400_with_an_error_body() {
        let app = app_with(ScriptedModel::replying("x"));
        let request = Request::builder()
            .method("POST")
            .uri(ROUTE_CHAT)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from("{not json"))
            .unwrap();
        let (status, text) = send(&app, request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn missing_content_type_is_a_415() {
        let app = app_with(ScriptedModel::replying("x"));
        let request = Request::builder()
            .method("POST")
            .uri(ROUTE_CHAT)
            .body(Body::from(json!({"messages": [user("hi")]}).to_string()))
            .unwrap();
        let (status, text) = send(&app, request).await;
        assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let body: Value = serde_json::from_str(&text).unwrap();
        assert!(body["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn unknown_fields_are_ignored() {
        let app = app_with(ScriptedModel::replying("ok"));
        let (status, _) = chat(
            &app,
            json!({"messages": [user("hi")], "logprobs": true, "n": 1, "model": "whatever"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn too_many_stop_sequences_are_rejected() {
        let app = app_with(ScriptedModel::replying("x"));
        let (status, _) = chat(
            &app,
            json!({"messages": [user("hi")], "stop": ["a", "b", "c", "d", "e"]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn hitting_the_token_limit_reports_length() {
        let mut model = ScriptedModel::replying("cut off");
        model.outcome.finish = FinishReason::Length;
        let app = app_with(model);
        let (_, body) = chat(&app, json!({"messages": [user("hi")], "max_tokens": 2})).await;
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["choices"][0]["message"]["content"], "cut off");
    }

    // ---- tool calls ------------------------------------------------------------------------

    #[tokio::test]
    async fn a_tool_call_reply_has_tool_calls_and_null_content() {
        let mut model = ScriptedModel::replying("");
        model.chat_text.clear();
        model.outcome = tool_outcome(None);
        let app = app_with(model);

        let (status, body) = chat(
            &app,
            json!({
                "messages": [user("2+2?")],
                "tools": [{"type": "function", "function": {"name": "calculator", "parameters": {"type": "object"}}}]
            }),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"], Value::Null);
        assert_eq!(message["tool_calls"][0]["id"], "call_7");
        assert_eq!(message["tool_calls"][0]["type"], "function");
        assert_eq!(message["tool_calls"][0]["function"]["name"], "calculator");
        let arguments: Value = serde_json::from_str(
            message["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(arguments, json!({"expression": "2+2"}));
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    }

    #[tokio::test]
    async fn text_before_a_tool_call_is_kept_as_content() {
        let mut model = ScriptedModel::replying("");
        model.outcome = tool_outcome(Some("Let me check."));
        let app = app_with(model);
        let (_, body) = chat(&app, json!({"messages": [user("2+2?")]})).await;
        assert_eq!(body["choices"][0]["message"]["content"], "Let me check.");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    }

    /// Records every conversation that reaches the model.
    struct Recording {
        inner: ScriptedModel,
        seen: Arc<std::sync::Mutex<Vec<ChatInput>>>,
    }

    impl LanguageModel for Recording {
        fn run(
            &mut self,
            prompt: String,
            config: djinn_core::lm::config::RunConfig,
        ) -> std::pin::Pin<Box<dyn futures::Stream<Item = djinn_models::Result<String>> + Send + '_>>
        {
            self.inner.run(prompt, config)
        }

        fn chat(&mut self, input: ChatInput) -> ChatStream<'_> {
            self.seen.lock().unwrap().push(input.clone());
            self.inner.chat(input)
        }
    }

    fn recording(inner: ScriptedModel) -> (Recording, Arc<std::sync::Mutex<Vec<ChatInput>>>) {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Recording {
                inner,
                seen: seen.clone(),
            },
            seen,
        )
    }

    #[tokio::test]
    async fn a_tool_result_follow_up_reaches_the_model_and_gets_an_answer() {
        let (model, seen) = recording(ScriptedModel::replying("It is 4."));
        let app = app_with(model);

        let (status, body) = chat(
            &app,
            json!({
                "messages": [
                    user("2+2?"),
                    {"role": "assistant", "content": null, "tool_calls": [{
                        "id": "call_7", "type": "function",
                        "function": {"name": "calculator", "arguments": "{\"expression\":\"2+2\"}"}
                    }]},
                    {"role": "tool", "tool_call_id": "call_7", "content": "4"}
                ]
            }),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["choices"][0]["message"]["content"], "It is 4.");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");

        let seen = seen.lock().unwrap();
        let roles: Vec<Role> = seen[0].messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, vec![Role::User, Role::Assistant, Role::Tool]);
        assert_eq!(seen[0].messages[2].tool_call_id.as_deref(), Some("call_7"));
    }

    #[tokio::test]
    async fn a_tool_message_without_a_matching_call_is_rejected() {
        let app = app_with(ScriptedModel::replying("x"));
        let (status, body) = chat(
            &app,
            json!({"messages": [user("hi"), {"role": "tool", "tool_call_id": "nope", "content": "4"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"].as_str().unwrap().contains("nope"));
    }

    #[tokio::test]
    async fn declared_tools_and_options_reach_the_model() {
        let (model, seen) = recording(ScriptedModel::replying("ok"));
        let app = app_with(model);

        chat(
            &app,
            json!({
                "messages": [user("hi")],
                "tools": [{"type": "function", "function": {"name": "calculator", "description": "math", "parameters": {"type": "object"}}}],
                "max_tokens": 33, "temperature": 0.25, "top_p": 0.5, "seed": 9,
                "stop": "END"
            }),
        )
        .await;

        let seen = seen.lock().unwrap();
        let input = &seen[0];
        assert_eq!(input.tools[0].function.name, "calculator");
        assert_eq!(input.params.max_tokens, Some(33));
        assert_eq!(input.params.temperature, Some(0.25));
        assert_eq!(input.params.top_p, Some(0.5));
        assert_eq!(input.params.seed, Some(9));
        assert_eq!(input.params.stop, vec!["END"]);
    }

    #[tokio::test]
    async fn stop_given_as_a_list_reaches_the_model_as_a_list() {
        let (model, seen) = recording(ScriptedModel::replying("ok"));
        let app = app_with(model);
        chat(&app, json!({"messages": [user("hi")], "stop": ["a", "b"]})).await;
        assert_eq!(seen.lock().unwrap()[0].params.stop, vec!["a", "b"]);
    }

    // ---- models ----------------------------------------------------------------------------

    #[tokio::test]
    async fn models_lists_the_loaded_model() {
        let app = app_with(ScriptedModel::replying("x"));
        let request = Request::builder()
            .uri(ROUTE_MODELS)
            .body(Body::empty())
            .unwrap();
        let (status, text) = send(&app, request).await;
        assert_eq!(status, StatusCode::OK);
        let body: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"].as_array().unwrap().len(), 1);
        assert_eq!(body["data"][0]["id"], "test-model");
    }

    // ---- errors ----------------------------------------------------------------------------

    struct ErrorModel(fn() -> djinn_models::Error);

    impl LanguageModel for ErrorModel {
        fn run(
            &mut self,
            _prompt: String,
            _config: djinn_core::lm::config::RunConfig,
        ) -> std::pin::Pin<Box<dyn futures::Stream<Item = djinn_models::Result<String>> + Send + '_>>
        {
            Box::pin(stream::empty())
        }

        fn chat(&mut self, _input: ChatInput) -> ChatStream<'_> {
            let error = (self.0)();
            Box::pin(stream::once(async move { Err(error) }))
        }
    }

    fn too_long() -> djinn_models::Error {
        djinn_models::Error::ContextLengthExceeded {
            prompt_tokens: 40000,
            context_length: 32768,
        }
    }

    #[tokio::test]
    async fn a_conversation_that_is_too_long_is_a_400_with_a_code() {
        let app = app_with(ErrorModel(too_long));
        let (status, body) = chat(&app, json!({"messages": [user("hi")]})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        assert!(body["error"]["message"].as_str().unwrap().contains("32768"));
    }

    #[tokio::test]
    async fn a_too_long_conversation_is_a_real_400_even_when_streaming() {
        let app = app_with(ErrorModel(too_long));
        let (status, text) = send(
            &app,
            post(&json!({"messages": [user("hi")], "stream": true})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["error"]["code"], "context_length_exceeded");
    }

    #[tokio::test]
    async fn an_internal_failure_is_a_500_without_details() {
        let app = app_with(ScriptedModel::failing());
        let (status, body) = chat(&app, json!({"messages": [user("hi")]})).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let message = body["error"]["message"].as_str().unwrap();
        assert!(
            !message.contains("model error"),
            "leaked details: {message}"
        );
    }

    #[tokio::test]
    async fn a_model_without_chat_support_is_a_501() {
        let app = app_with(ScriptedModel::completing(&["a"]));
        let (status, body) = chat(&app, json!({"messages": [user("hi")]})).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(body["error"]["code"], "chat_not_supported");
    }

    // ---- streaming -------------------------------------------------------------------------

    #[tokio::test]
    async fn text_streams_as_chunks_then_finish_then_done() {
        let mut model = ScriptedModel::replying("");
        model.chat_text = vec!["Hi".into(), " there".into()];
        model.outcome.content = Some("Hi there".into());
        let app = app_with(model);

        let (status, body) = send(
            &app,
            post(&json!({"messages": [user("hi")], "stream": true})),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        let data = sse_data(&body);
        assert_eq!(data.last().unwrap(), "[DONE]");

        let chunks = chunks(&body);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert!(chunks
            .iter()
            .all(|c| c["object"] == "chat.completion.chunk"));
        assert_eq!(content_deltas(&chunks), vec!["Hi", " there"]);

        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["usage"]["total_tokens"], 8);
        assert!(
            chunks[..chunks.len() - 1]
                .iter()
                .all(|c| c["choices"][0]["finish_reason"].is_null()),
            "only the last chunk carries a finish reason"
        );
    }

    #[tokio::test]
    async fn the_streaming_response_is_an_event_stream() {
        let app = app_with(ScriptedModel::replying("x"));
        let response = app
            .oneshot(post(&json!({"messages": [user("hi")], "stream": true})))
            .await
            .unwrap();
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
    }

    #[tokio::test]
    async fn a_streamed_tool_call_arrives_complete_in_the_final_chunk_only() {
        let mut model = ScriptedModel::replying("");
        model.chat_text = vec!["Let me check.".into()];
        model.outcome = tool_outcome(Some("Let me check."));
        let app = app_with(model);

        let (_, body) = send(
            &app,
            post(&json!({"messages": [user("2+2?")], "stream": true})),
        )
        .await;

        let chunks = chunks(&body);
        assert_eq!(content_deltas(&chunks), vec!["Let me check."]);
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(chunk["choices"][0]["delta"].get("tool_calls").is_none());
        }
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
        let call = &last["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["id"], "call_7");
        assert_eq!(call["function"]["name"], "calculator");
        assert_eq!(call["function"]["arguments"], r#"{"expression":"2+2"}"#);
        assert!(!body.contains("<tool_call>"));
    }

    #[tokio::test]
    async fn a_tool_call_only_reply_streams_with_just_the_final_chunk() {
        let mut model = ScriptedModel::replying("");
        model.chat_text.clear();
        model.outcome = tool_outcome(None);
        let app = app_with(model);
        let (status, body) = send(
            &app,
            post(&json!({"messages": [user("2+2?")], "stream": true})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let chunks = chunks(&body);
        assert!(content_deltas(&chunks).is_empty());
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
        assert_eq!(sse_data(&body).last().unwrap(), "[DONE]");
    }

    struct FailsMidStream;

    impl LanguageModel for FailsMidStream {
        fn run(
            &mut self,
            _prompt: String,
            _config: djinn_core::lm::config::RunConfig,
        ) -> std::pin::Pin<Box<dyn futures::Stream<Item = djinn_models::Result<String>> + Send + '_>>
        {
            Box::pin(stream::empty())
        }

        fn chat(&mut self, _input: ChatInput) -> ChatStream<'_> {
            Box::pin(stream::iter(vec![
                Ok(ChatEvent::Text("partial".into())),
                Err(djinn_models::Error::Anyhow(anyhow::anyhow!("boom"))),
            ]))
        }
    }

    #[tokio::test]
    async fn an_error_mid_stream_sends_an_error_object_and_no_done() {
        let app = app_with(FailsMidStream);
        let (status, body) = send(
            &app,
            post(&json!({"messages": [user("hi")], "stream": true})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "headers were already sent");
        let data = sse_data(&body);
        assert!(!data.contains(&"[DONE]".to_owned()));
        let last: Value = serde_json::from_str(data.last().unwrap()).unwrap();
        assert!(last["error"]["message"].is_string());
        assert!(!last["error"]["message"].as_str().unwrap().contains("boom"));
        assert!(body.contains("partial"));
    }

    // ---- cancellation ----------------------------------------------------------------------

    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Streams text forever, and records when the stream is dropped.
    struct EndlessModel {
        dropped: Arc<AtomicBool>,
    }

    impl LanguageModel for EndlessModel {
        fn run(
            &mut self,
            _prompt: String,
            _config: djinn_core::lm::config::RunConfig,
        ) -> std::pin::Pin<Box<dyn futures::Stream<Item = djinn_models::Result<String>> + Send + '_>>
        {
            Box::pin(stream::empty())
        }

        fn chat(&mut self, _input: ChatInput) -> ChatStream<'_> {
            let guard = DropFlag(self.dropped.clone());
            Box::pin(stream::unfold(guard, |guard| async move {
                tokio::time::sleep(Duration::from_millis(5)).await;
                Some((Ok(ChatEvent::Text("x".into())), guard))
            }))
        }
    }

    #[tokio::test]
    async fn a_client_that_disconnects_mid_stream_releases_the_model() {
        let dropped = Arc::new(AtomicBool::new(false));
        let app = app_with(EndlessModel {
            dropped: dropped.clone(),
        });

        let response = app
            .clone()
            .oneshot(post(&json!({"messages": [user("hi")], "stream": true})))
            .await
            .unwrap();
        let mut body = response.into_body();
        // Read the opening chunk, then walk away.
        let first = body.frame().await.unwrap().unwrap();
        assert!(first.is_data());
        drop(body);

        // The model must become available again quickly: raw completion needs the same lock.
        let request = Request::builder()
            .method("POST")
            .uri("/complete")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({"prompt": "hi"}).to_string()))
            .unwrap();
        let (status, _) = tokio::time::timeout(Duration::from_secs(5), send(&app, request))
            .await
            .expect("the model lock was never released");

        assert_eq!(status, StatusCode::OK);
        assert!(
            dropped.load(Ordering::SeqCst),
            "the abandoned generation was not stopped"
        );
    }
}
