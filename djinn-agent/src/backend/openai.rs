//! A backend for any server that speaks the OpenAI chat-completions protocol.

use std::sync::atomic::{AtomicU64, Ordering};

use async_stream::try_stream;
use djinn_core::chat::{ChatRequest, Delta, FinishReason, ToolCall};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;

use super::sse::SseParser;
use super::{Backend, BackendError, BackendEvent, ChatResult};

/// Where the djinn server listens by default.
pub const DEFAULT_BASE_URL: &str = "http://localhost:8080/v1";
/// Model name sent when none is configured; the djinn server ignores it.
pub const DEFAULT_MODEL: &str = "default";

/// Talks to `{base_url}/chat/completions` with streaming responses.
pub struct OpenAiBackend {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl OpenAiBackend {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            api_key: api_key.filter(|key| !key.is_empty()),
        }
    }

    /// The URL chat requests are sent to.
    pub fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// The URL of the model list, used to check that the server is up.
    pub fn models_url(&self) -> String {
        format!("{}/models", self.base_url.trim_end_matches('/'))
    }

    /// Whether the server answers at all (any HTTP status counts as reachable).
    pub async fn check_reachable(&self) -> Result<(), BackendError> {
        let url = self.models_url();
        let mut request = self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(5));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        request
            .send()
            .await
            .map(|_| ())
            .map_err(|error| connection_error(&url, &error))
    }
}

impl Default for OpenAiBackend {
    fn default() -> Self {
        Self::new(DEFAULT_BASE_URL, DEFAULT_MODEL, None)
    }
}

fn connection_error(url: &str, error: &reqwest::Error) -> BackendError {
    // reqwest's Display is terse ("error sending request"); the cause chain says why.
    let mut message = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    BackendError::Connection {
        url: url.to_owned(),
        message,
    }
}

/// Picks the human-readable message out of an error response body.
fn error_message(body: &str) -> String {
    let from_json =
        serde_json::from_str::<Value>(body)
            .ok()
            .and_then(|value| match value.get("error") {
                Some(Value::String(message)) => Some(message.clone()),
                Some(error @ Value::Object(_)) => error
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                _ => value
                    .get("message")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            });
    let message = from_json.unwrap_or_else(|| body.trim().to_owned());
    if message.is_empty() {
        return "(no error message)".to_owned();
    }
    let mut message = message;
    if message.chars().count() > 300 {
        message = message.chars().take(300).collect::<String>() + "...";
    }
    message
}

impl Backend for OpenAiBackend {
    fn chat(&self, mut request: ChatRequest) -> BoxStream<'_, Result<BackendEvent, BackendError>> {
        request.model = Some(self.model.clone());
        request.stream = Some(true);
        let url = self.completions_url();

        Box::pin(try_stream! {
            let mut builder = self.client.post(&url).json(&request);
            if let Some(key) = &self.api_key {
                builder = builder.bearer_auth(key);
            }
            let response = builder.send().await.map_err(|e| connection_error(&url, &e))?;

            let status = response.status();
            let response = if status.is_success() {
                response
            } else {
                let body = response.text().await.unwrap_or_default();
                Err(BackendError::Status { status: status.as_u16(), message: error_message(&body) })?;
                unreachable!("the error above ends the stream");
            };

            let mut parser = SseParser::new();
            let mut assembler = Assembler::default();
            let mut bytes = response.bytes_stream();

            'read: while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(|_| BackendError::StreamEndedEarly)?;
                for event in parser.push(&chunk) {
                    match assembler.accept(&event.data)? {
                        Accepted::Nothing => {}
                        Accepted::Text(text) => yield BackendEvent::Text(text),
                        Accepted::Done => break 'read,
                    }
                }
            }
            if !assembler.is_done() {
                if let Some(event) = parser.finish() {
                    if let Accepted::Text(text) = assembler.accept(&event.data)? {
                        yield BackendEvent::Text(text);
                    }
                }
            }

            yield BackendEvent::Done(assembler.finish()?);
        })
    }
}

// ---- stream assembly ----------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Accepted {
    Nothing,
    Text(String),
    Done,
}

/// What a chunk carries, leniently: other servers omit fields djinn's own chunks always have.
#[derive(Deserialize)]
struct RawChunk {
    #[serde(default)]
    choices: Vec<RawChoice>,
}

#[derive(Deserialize)]
struct RawChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Default)]
struct PartialCall {
    id: Option<String>,
    name: String,
    arguments: String,
}

/// Builds a [`ChatResult`] from streamed chunks: concatenates text, and merges tool-call fragments
/// (which may arrive one piece at a time, keyed by `index`) into complete calls.
#[derive(Default)]
pub(crate) struct Assembler {
    content: String,
    calls: Vec<PartialCall>,
    finish: Option<FinishReason>,
    done: bool,
}

fn next_call_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!("call_{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

impl Assembler {
    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    /// Takes the payload of one SSE event.
    pub(crate) fn accept(&mut self, data: &str) -> Result<Accepted, BackendError> {
        let data = data.trim();
        if data == "[DONE]" {
            self.done = true;
            return Ok(Accepted::Done);
        }
        if data.is_empty() {
            return Ok(Accepted::Nothing);
        }

        let value: Value = serde_json::from_str(data)
            .map_err(|e| BackendError::Malformed(format!("invalid JSON in stream: {e}")))?;
        if let Some(error) = value.get("error") {
            let message = match error {
                Value::String(message) => message.clone(),
                Value::Object(_) => error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
                    .to_owned(),
                other => other.to_string(),
            };
            return Err(BackendError::Server(message));
        }
        let chunk: RawChunk = serde_json::from_value(value)
            .map_err(|e| BackendError::Malformed(format!("unexpected chunk shape: {e}")))?;

        let mut text = String::new();
        for choice in chunk.choices {
            if let Some(content) = choice.delta.content {
                text.push_str(&content);
            }
            for call in choice.delta.tool_calls.into_iter().flatten() {
                self.merge_call(call);
            }
            if let Some(reason) = choice.finish_reason {
                self.finish = Some(match reason.as_str() {
                    "length" => FinishReason::Length,
                    "tool_calls" | "function_call" => FinishReason::ToolCalls,
                    _ => FinishReason::Stop,
                });
            }
        }

        if text.is_empty() {
            Ok(Accepted::Nothing)
        } else {
            self.content.push_str(&text);
            Ok(Accepted::Text(text))
        }
    }

    fn merge_call(&mut self, delta: djinn_core::chat::ToolCallDelta) {
        // Without an index, a fragment carrying an id starts a new call and one without an id
        // continues the latest call.
        let index = match delta.index {
            Some(index) => index as usize,
            None if delta.id.is_some() => self.calls.len(),
            None => self.calls.len().saturating_sub(1),
        };
        while self.calls.len() <= index {
            self.calls.push(PartialCall::default());
        }
        let call = &mut self.calls[index];
        if let Some(id) = delta.id {
            call.id = Some(id);
        }
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                if call.name.is_empty() {
                    call.name = name;
                }
            }
            if let Some(arguments) = function.arguments {
                call.arguments.push_str(&arguments);
            }
        }
    }

    /// Ends the stream. Fails if the server never signalled completion.
    pub(crate) fn finish(self) -> Result<ChatResult, BackendError> {
        if !self.done && self.finish.is_none() {
            return Err(BackendError::StreamEndedEarly);
        }

        let mut tool_calls = Vec::with_capacity(self.calls.len());
        for call in self.calls {
            if call.name.is_empty() {
                return Err(BackendError::Malformed(
                    "a tool call has no function name".to_owned(),
                ));
            }
            let arguments = if call.arguments.trim().is_empty() {
                "{}".to_owned()
            } else {
                match serde_json::from_str::<Value>(&call.arguments) {
                    Ok(Value::Object(_)) => call.arguments,
                    Ok(_) => {
                        return Err(BackendError::Malformed(format!(
                            "arguments of tool call `{}` are not a JSON object",
                            call.name
                        )))
                    }
                    Err(error) => {
                        return Err(BackendError::Malformed(format!(
                            "arguments of tool call `{}` are not valid JSON: {error}",
                            call.name
                        )))
                    }
                }
            };
            tool_calls.push(ToolCall::function(
                call.id.unwrap_or_else(next_call_id),
                call.name,
                arguments,
            ));
        }

        let finish = if tool_calls.is_empty() {
            self.finish.unwrap_or(FinishReason::Stop)
        } else {
            FinishReason::ToolCalls
        };
        Ok(ChatResult {
            content: (!self.content.is_empty()).then_some(self.content),
            tool_calls,
            finish,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use axum::body::{Body, Bytes};
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::Router;
    use djinn_core::chat::ChatMessage;
    use futures::stream;
    use serde_json::json;

    use super::*;

    // ---- assembler ---------------------------------------------------------------------------

    fn chunk(delta: Value, finish: Option<&str>) -> String {
        json!({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).to_string()
    }

    fn assemble(events: &[String]) -> Result<(String, ChatResult), BackendError> {
        let mut assembler = Assembler::default();
        let mut text = String::new();
        for event in events {
            if let Accepted::Text(t) = assembler.accept(event)? {
                text.push_str(&t);
            }
        }
        Ok((text, assembler.finish()?))
    }

    #[test]
    fn text_is_concatenated() {
        let (text, result) = assemble(&[
            chunk(json!({"role": "assistant", "content": ""}), None),
            chunk(json!({"content": "Good"}), None),
            chunk(json!({"content": " day"}), None),
            chunk(json!({}), Some("stop")),
            "[DONE]".into(),
        ])
        .unwrap();
        assert_eq!(text, "Good day");
        assert_eq!(result.content.as_deref(), Some("Good day"));
        assert_eq!(result.finish, FinishReason::Stop);
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn a_tool_call_in_one_chunk() {
        let (_, result) = assemble(&[
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                    "function": {"name": "calculator", "arguments": "{\"expression\":\"2+2\"}"}}]}),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
            "[DONE]".into(),
        ])
        .unwrap();
        assert_eq!(result.content, None);
        assert_eq!(result.finish, FinishReason::ToolCalls);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "call_a");
        assert_eq!(result.tool_calls[0].function.name, "calculator");
        assert_eq!(
            result.tool_calls[0].function.arguments,
            "{\"expression\":\"2+2\"}"
        );
    }

    #[test]
    fn fragmented_arguments_are_concatenated_and_parsed_once() {
        let (_, result) = assemble(&[
            chunk(json!({"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "calc", "arguments": ""}}]}), None),
            chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"expr"}}]}), None),
            chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "ession\":"}}]}), None),
            chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"1+1\"}"}}]}), None),
            chunk(json!({}), Some("tool_calls")),
            "[DONE]".into(),
        ])
        .unwrap();
        assert_eq!(
            result.tool_calls[0].function.arguments,
            "{\"expression\":\"1+1\"}"
        );
    }

    #[test]
    fn two_calls_are_kept_apart_by_index() {
        let (_, result) = assemble(&[
            chunk(json!({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "one", "arguments": "{}"}}]}), None),
            chunk(json!({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "two", "arguments": "{\"x\":"}}]}), None),
            chunk(json!({"tool_calls": [{"index": 1, "function": {"arguments": "1}"}}]}), None),
            "[DONE]".into(),
        ])
        .unwrap();
        let names: Vec<_> = result
            .tool_calls
            .iter()
            .map(|c| c.function.name.as_str())
            .collect();
        assert_eq!(names, vec!["one", "two"]);
        assert_eq!(result.tool_calls[1].function.arguments, "{\"x\":1}");
    }

    #[test]
    fn calls_without_indexes_are_told_apart_by_ids() {
        let (_, result) = assemble(&[
            chunk(json!({"tool_calls": [{"id": "a", "function": {"name": "one", "arguments": "{}"}}]}), None),
            chunk(json!({"tool_calls": [{"id": "b", "function": {"name": "two", "arguments": "{}"}}]}), None),
            "[DONE]".into(),
        ])
        .unwrap();
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].id, "a");
        assert_eq!(result.tool_calls[1].id, "b");
    }

    #[test]
    fn missing_ids_are_generated_and_unique() {
        let events = [
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"name": "one", "arguments": "{}"}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 1, "function": {"name": "two", "arguments": "{}"}}]}),
                None,
            ),
            "[DONE]".into(),
        ];
        let (_, first) = assemble(&events).unwrap();
        let (_, second) = assemble(&events).unwrap();
        let ids: Vec<&String> = first
            .tool_calls
            .iter()
            .chain(&second.tool_calls)
            .map(|c| &c.id)
            .collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 4, "{ids:?}");
        assert!(ids.iter().all(|id| !id.is_empty()));
    }

    #[test]
    fn empty_arguments_mean_no_arguments() {
        let (_, result) = assemble(&[
            chunk(json!({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "current_time"}}]}), None),
            "[DONE]".into(),
        ])
        .unwrap();
        assert_eq!(result.tool_calls[0].function.arguments, "{}");
    }

    #[test]
    fn invalid_argument_json_is_a_malformed_response() {
        let error = assemble(&[
            chunk(json!({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "calc", "arguments": "{\"x\": "}}]}), None),
            "[DONE]".into(),
        ])
        .unwrap_err();
        assert!(
            matches!(&error, BackendError::Malformed(m) if m.contains("calc") && m.contains("valid JSON")),
            "{error}"
        );
    }

    #[test]
    fn non_object_arguments_are_malformed() {
        let error = assemble(&[
            chunk(json!({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "calc", "arguments": "[1]"}}]}), None),
            "[DONE]".into(),
        ])
        .unwrap_err();
        assert!(matches!(error, BackendError::Malformed(_)));
    }

    #[test]
    fn a_call_without_a_name_is_malformed() {
        let error = assemble(&[
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "a", "function": {"arguments": "{}"}}]}),
                None,
            ),
            "[DONE]".into(),
        ])
        .unwrap_err();
        assert!(matches!(error, BackendError::Malformed(m) if m.contains("name")));
    }

    #[test]
    fn a_stream_with_no_end_signal_ended_early() {
        let error = assemble(&[chunk(json!({"content": "partial"}), None)]).unwrap_err();
        assert_eq!(error, BackendError::StreamEndedEarly);
    }

    #[test]
    fn a_finish_reason_alone_completes_the_stream() {
        // Some servers close the connection without sending [DONE].
        let (text, result) = assemble(&[
            chunk(json!({"content": "ok"}), None),
            chunk(json!({}), Some("stop")),
        ])
        .unwrap();
        assert_eq!(text, "ok");
        assert_eq!(result.finish, FinishReason::Stop);
    }

    #[test]
    fn tool_calls_force_the_tool_calls_finish_reason() {
        let (_, result) = assemble(&[
            chunk(json!({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "x", "arguments": "{}"}}]}), Some("stop")),
            "[DONE]".into(),
        ])
        .unwrap();
        assert_eq!(result.finish, FinishReason::ToolCalls);
    }

    #[test]
    fn length_and_unknown_finish_reasons() {
        let (_, length) = assemble(&[
            chunk(json!({"content": "x"}), Some("length")),
            "[DONE]".into(),
        ])
        .unwrap();
        assert_eq!(length.finish, FinishReason::Length);
        let (_, other) = assemble(&[
            chunk(json!({"content": "x"}), Some("content_filter")),
            "[DONE]".into(),
        ])
        .unwrap();
        assert_eq!(other.finish, FinishReason::Stop);
    }

    #[test]
    fn null_content_and_empty_choices_are_tolerated() {
        let (text, _) = assemble(&[
            chunk(json!({"content": null}), None),
            json!({"choices": [], "usage": {"total_tokens": 3}}).to_string(),
            chunk(json!({"content": "hi"}), None),
            "[DONE]".into(),
        ])
        .unwrap();
        assert_eq!(text, "hi");
    }

    #[test]
    fn an_error_object_in_the_stream_is_reported() {
        let mut assembler = Assembler::default();
        let error = assembler
            .accept(r#"{"error": {"message": "model exploded", "type": "server_error"}}"#)
            .unwrap_err();
        assert_eq!(error, BackendError::Server("model exploded".into()));
    }

    #[test]
    fn garbage_in_the_stream_is_malformed() {
        let mut assembler = Assembler::default();
        assert!(matches!(
            assembler.accept("not json at all").unwrap_err(),
            BackendError::Malformed(_)
        ));
    }

    #[test]
    fn error_messages_are_extracted_from_the_usual_shapes() {
        assert_eq!(
            error_message(r#"{"error":{"message":"bad things"}}"#),
            "bad things"
        );
        assert_eq!(error_message(r#"{"error":"ollama style"}"#), "ollama style");
        assert_eq!(error_message(r#"{"message":"flat"}"#), "flat");
        assert_eq!(error_message("plain text body\n"), "plain text body");
        assert_eq!(error_message(""), "(no error message)");
        assert!(error_message(&"x".repeat(1000)).len() < 400);
    }

    // ---- over HTTP, against a local fake server -------------------------------------------------

    #[derive(Default)]
    struct Captured {
        bodies: Vec<Value>,
        headers: Vec<HeaderMap>,
    }

    type Shared = Arc<Mutex<Captured>>;

    /// What the fake server answers to a chat request.
    #[derive(Clone)]
    enum Respond {
        /// An event stream sent in these byte chunks, with a short pause between them.
        Sse(Vec<Vec<u8>>),
        Status(StatusCode, String),
    }

    async fn fake_chat(
        State((captured, respond)): State<(Shared, Respond)>,
        headers: HeaderMap,
        body: Bytes,
    ) -> Response {
        {
            let mut captured = captured.lock().unwrap();
            captured.bodies.push(serde_json::from_slice(&body).unwrap());
            captured.headers.push(headers);
        }
        match respond {
            Respond::Status(status, body) => (status, body).into_response(),
            Respond::Sse(chunks) => {
                let stream = stream::unfold(chunks.into_iter(), |mut chunks| async move {
                    let chunk = chunks.next()?;
                    tokio::time::sleep(Duration::from_millis(2)).await;
                    Some((
                        Ok::<_, std::convert::Infallible>(Bytes::from(chunk)),
                        chunks,
                    ))
                });
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }
        }
    }

    async fn serve(respond: Respond) -> (String, Shared) {
        let captured = Shared::default();
        let app = Router::new()
            .route("/v1/chat/completions", post(fake_chat))
            .route("/v1/models", get(|| async { "ok" }))
            .with_state((captured.clone(), respond));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{address}/v1"), captured)
    }

    fn sse(events: &[String]) -> Vec<u8> {
        events
            .iter()
            .map(|e| format!("data: {e}\n\n"))
            .collect::<String>()
            .into_bytes()
    }

    fn simple_request() -> ChatRequest {
        ChatRequest {
            messages: vec![ChatMessage::user("hi")],
            ..ChatRequest::default()
        }
    }

    async fn collect(backend: &OpenAiBackend) -> (Vec<BackendEvent>, Option<BackendError>) {
        let mut stream = backend.chat(simple_request());
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => events.push(event),
                Err(error) => return (events, Some(error)),
            }
        }
        (events, None)
    }

    fn text_events(events: &[BackendEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|e| match e {
                BackendEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn streams_text_then_a_result() {
        let body = sse(&[
            chunk(json!({"role": "assistant", "content": ""}), None),
            chunk(json!({"content": "Hi"}), None),
            chunk(json!({"content": " there"}), None),
            chunk(json!({}), Some("stop")),
            "[DONE]".into(),
        ]);
        let (url, _) = serve(Respond::Sse(vec![body])).await;
        let backend = OpenAiBackend::new(url, "m", None);

        let (events, error) = collect(&backend).await;
        assert_eq!(error, None);
        assert_eq!(text_events(&events), vec!["Hi", " there"]);
        match events.last().unwrap() {
            BackendEvent::Done(result) => {
                assert_eq!(result.content.as_deref(), Some("Hi there"));
                assert_eq!(result.finish, FinishReason::Stop);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn sends_the_model_and_asks_for_streaming() {
        let (url, captured) = serve(Respond::Sse(vec![sse(&[
            chunk(json!({"content": "x"}), Some("stop")),
            "[DONE]".into(),
        ])]))
        .await;
        let backend = OpenAiBackend::new(url, "my-model", None);
        collect(&backend).await;

        let captured = captured.lock().unwrap();
        assert_eq!(captured.bodies[0]["model"], "my-model");
        assert_eq!(captured.bodies[0]["stream"], true);
        assert_eq!(captured.bodies[0]["messages"][0]["content"], "hi");
    }

    #[tokio::test]
    async fn the_bearer_token_is_sent_only_when_configured() {
        let ok = || {
            sse(&[
                chunk(json!({"content": "x"}), Some("stop")),
                "[DONE]".into(),
            ])
        };

        let (url, captured) = serve(Respond::Sse(vec![ok()])).await;
        collect(&OpenAiBackend::new(url, "m", Some("secret".into()))).await;
        assert_eq!(
            captured.lock().unwrap().headers[0]["authorization"],
            "Bearer secret"
        );

        let (url, captured) = serve(Respond::Sse(vec![ok()])).await;
        collect(&OpenAiBackend::new(url, "m", None)).await;
        assert!(captured.lock().unwrap().headers[0]
            .get("authorization")
            .is_none());

        let (url, captured) = serve(Respond::Sse(vec![ok()])).await;
        collect(&OpenAiBackend::new(url, "m", Some(String::new()))).await;
        assert!(
            captured.lock().unwrap().headers[0]
                .get("authorization")
                .is_none(),
            "an empty key is no key"
        );
    }

    #[tokio::test]
    async fn events_split_across_network_chunks_are_reassembled() {
        let whole = sse(&[
            chunk(json!({"content": "\u{e9}t\u{e9}"}), None),
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "calc", "arguments": "{\"a\":1}"}}]}),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
            "[DONE]".into(),
        ]);
        // One byte per network chunk is the worst case.
        let chunks: Vec<Vec<u8>> = whole.iter().map(|b| vec![*b]).collect();
        let (url, _) = serve(Respond::Sse(chunks)).await;

        let (events, error) = collect(&OpenAiBackend::new(url, "m", None)).await;
        assert_eq!(error, None);
        assert_eq!(text_events(&events), vec!["\u{e9}t\u{e9}"]);
        let BackendEvent::Done(result) = events.last().unwrap() else {
            panic!("no Done")
        };
        assert_eq!(result.tool_calls[0].function.arguments, "{\"a\":1}");
        assert_eq!(result.finish, FinishReason::ToolCalls);
    }

    #[tokio::test]
    async fn an_error_status_carries_the_servers_message() {
        let (url, _) = serve(Respond::Status(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"context too long","code":"context_length_exceeded"}}"#.into(),
        ))
        .await;
        let (events, error) = collect(&OpenAiBackend::new(url, "m", None)).await;
        assert!(events.is_empty());
        let error = error.unwrap();
        assert_eq!(
            error,
            BackendError::Status {
                status: 400,
                message: "context too long".into()
            }
        );
        let shown = error.to_string();
        assert!(
            shown.contains("400") && shown.contains("context too long"),
            "{shown}"
        );
    }

    #[tokio::test]
    async fn a_non_json_error_body_is_shown_as_is() {
        let (url, _) = serve(Respond::Status(
            StatusCode::BAD_GATEWAY,
            "upstream down".into(),
        ))
        .await;
        let (_, error) = collect(&OpenAiBackend::new(url, "m", None)).await;
        assert_eq!(
            error,
            Some(BackendError::Status {
                status: 502,
                message: "upstream down".into()
            })
        );
    }

    #[tokio::test]
    async fn a_stream_cut_off_before_the_end_keeps_the_text_already_received() {
        let body = sse(&[chunk(json!({"content": "partial answer"}), None)]);
        let (url, _) = serve(Respond::Sse(vec![body])).await;

        let (events, error) = collect(&OpenAiBackend::new(url, "m", None)).await;
        assert_eq!(text_events(&events), vec!["partial answer"]);
        assert_eq!(error, Some(BackendError::StreamEndedEarly));
        assert!(error.unwrap().to_string().contains("ended before"));
    }

    #[tokio::test]
    async fn an_error_inside_the_stream_is_reported() {
        let body = sse(&[
            chunk(json!({"content": "so far"}), None),
            json!({"error": {"message": "Something went wrong D:", "type": "server_error"}})
                .to_string(),
        ]);
        let (url, _) = serve(Respond::Sse(vec![body])).await;
        let (events, error) = collect(&OpenAiBackend::new(url, "m", None)).await;
        assert_eq!(text_events(&events), vec!["so far"]);
        assert_eq!(
            error,
            Some(BackendError::Server("Something went wrong D:".into()))
        );
    }

    #[tokio::test]
    async fn a_missing_trailing_blank_line_does_not_lose_the_last_event() {
        let mut body = sse(&[chunk(json!({"content": "hi"}), Some("stop"))]);
        body.extend_from_slice(b"data: [DONE]"); // no final \n\n
        let (url, _) = serve(Respond::Sse(vec![body])).await;
        let (events, error) = collect(&OpenAiBackend::new(url, "m", None)).await;
        assert_eq!(error, None);
        assert!(matches!(events.last(), Some(BackendEvent::Done(_))));
    }

    #[tokio::test]
    async fn nothing_listening_names_the_url_that_was_tried() {
        // Bind then drop to get a port that is almost certainly free.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let url = format!("http://127.0.0.1:{port}/v1");

        let (events, error) = collect(&OpenAiBackend::new(&url, "m", None)).await;
        assert!(events.is_empty());
        let error = error.unwrap();
        match &error {
            BackendError::Connection { url: tried, .. } => {
                assert_eq!(tried, &format!("{url}/chat/completions"));
            }
            other => panic!("expected a connection error, got {other:?}"),
        }
        assert!(
            error.to_string().contains("could not connect to"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn trailing_slashes_in_the_base_url_are_fine() {
        let backend = OpenAiBackend::new("http://example.test/v1/", "m", None);
        assert_eq!(
            backend.completions_url(),
            "http://example.test/v1/chat/completions"
        );
        assert_eq!(backend.models_url(), "http://example.test/v1/models");
    }

    #[tokio::test]
    async fn reachability_check() {
        let (url, _) = serve(Respond::Sse(vec![])).await;
        assert!(OpenAiBackend::new(url, "m", None)
            .check_reachable()
            .await
            .is_ok());

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let error = OpenAiBackend::new(format!("http://127.0.0.1:{port}/v1"), "m", None)
            .check_reachable()
            .await
            .unwrap_err();
        assert!(matches!(error, BackendError::Connection { .. }));
    }

    #[test]
    fn defaults_match_the_djinn_server() {
        let backend = OpenAiBackend::default();
        assert_eq!(
            backend.completions_url(),
            "http://localhost:8080/v1/chat/completions"
        );
        assert_eq!(backend.model, "default");
    }
}
