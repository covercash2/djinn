//! Wire types for an OpenAI-compatible chat completions API.
//!
//! Shared by `djinn-server` (which serves the API) and `djinn-agent` (which consumes it) so
//! the two cannot drift apart. Only the subset of the protocol that djinn supports is modelled;
//! unknown request fields are ignored on deserialization.
//!
//! Conventions that follow the OpenAI protocol and are easy to get wrong:
//!
//! - `content` is nullable: an assistant message that only carries tool calls has `null` content.
//! - [`FunctionCall::arguments`] is a **string** holding JSON, not a JSON object.
//! - `stop` may be a single string or a list of strings ([`StopSpec`]).

use serde::{Deserialize, Serialize};

mod validate;

pub use validate::{validate_request, ChatRequestError, MAX_STOP_SEQUENCES};

/// Who a message is from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// The kind of a tool. Only functions exist in the supported protocol subset.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum ToolKind {
    #[default]
    Function,
}

/// A function the model may call, as described to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FunctionSpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing the arguments object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<Object>))]
    pub parameters: Option<serde_json::Value>,
}

/// A tool declared in a request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ToolSpec {
    #[serde(rename = "type", default)]
    pub kind: ToolKind,
    pub function: FunctionSpec,
}

impl ToolSpec {
    /// Convenience constructor for a function tool.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            kind: ToolKind::Function,
            function: FunctionSpec {
                name: name.into(),
                description: Some(description.into()),
                parameters: Some(parameters),
            },
        }
    }
}

/// A function call chosen by the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FunctionCall {
    pub name: String,
    /// The arguments object, **encoded as a JSON string**.
    pub arguments: String,
}

/// A tool call in a complete (non-delta) message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: ToolKind,
    pub function: FunctionCall,
}

impl ToolCall {
    /// Convenience constructor for a function call.
    pub fn function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: ToolKind::Function,
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

/// One message of a conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ChatMessage {
    pub role: Role,
    /// Message text; `null` for an assistant message that only carries tool calls.
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls made by the assistant in this message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For `tool` messages: the id of the call this message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::text(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::text(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(Role::Assistant, content)
    }

    /// An assistant message that makes tool calls, with optional preceding text.
    pub fn assistant_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// The result of a tool call.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// `stop` as sent by clients: one string or a list of strings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

impl StopSpec {
    /// The stop strings as a list.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSpec::One(s) => vec![s],
            StopSpec::Many(v) => v,
        }
    }

    /// The stop strings as a slice-backed list.
    pub fn to_vec(&self) -> Vec<String> {
        self.clone().into_vec()
    }
}

/// A chat completion request. Unknown fields are ignored.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ChatRequest {
    /// Accepted for compatibility; the server always uses the loaded model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSpec>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl ChatRequest {
    /// Whether the client asked for a streamed response.
    pub fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

/// Why generation ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// A natural end of turn or a stop sequence.
    Stop,
    /// The token limit was reached.
    Length,
    /// The model wants tool calls to be run.
    ToolCalls,
}

/// Token accounting for a response.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// One alternative in a [`ChatResponse`]. Only index 0 is ever produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
}

/// A complete (non-streamed) chat completion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ChatResponse {
    pub id: String,
    /// Always `chat.completion`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// Object type of a complete response.
pub const OBJECT_CHAT_COMPLETION: &str = "chat.completion";
/// Object type of a streamed chunk.
pub const OBJECT_CHAT_COMPLETION_CHUNK: &str = "chat.completion.chunk";

/// A function call as streamed: any field may be absent, and `arguments` may arrive in
/// fragments that must be concatenated per `index`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FunctionCallDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// A tool call as streamed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ToolCallDelta {
    /// Position of the call within the message; used to merge fragments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// The incremental part of a streamed message.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// One alternative in a [`ChatChunk`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    /// `null` until the final chunk.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
}

/// One server-sent event payload of a streamed response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ChatChunk {
    pub id: String,
    /// Always `chat.completion.chunk`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Token accounting; only present on the final chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One entry of [`ModelList`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ModelInfo {
    pub id: String,
    /// Always `model`.
    pub object: String,
    pub owned_by: String,
}

/// Response of `GET /v1/models`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ModelList {
    /// Always `list`.
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// The detail of an [`ErrorBody`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Error body, also sent as the payload of a streaming error event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

impl ErrorBody {
    pub fn new(kind: impl Into<String>, message: impl Into<String>, code: Option<&str>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                kind: kind.into(),
                code: code.map(str::to_owned),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse<T: serde::de::DeserializeOwned>(value: serde_json::Value) -> T {
        serde_json::from_value(value).expect("valid JSON for type")
    }

    #[test]
    fn minimal_request_deserializes() {
        let req: ChatRequest = parse(json!({"messages": [{"role": "user", "content": "hi"}]}));
        assert_eq!(req.messages, vec![ChatMessage::user("hi")]);
        assert!(!req.is_stream());
        assert!(req.tools.is_none() && req.stop.is_none() && req.model.is_none());
    }

    #[test]
    fn request_with_tools_and_options() {
        let req: ChatRequest = parse(json!({
            "model": "anything",
            "stream": true,
            "max_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 7,
            "messages": [{"role": "user", "content": "2+2?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate arithmetic",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
                }
            }]
        }));
        assert!(req.is_stream());
        assert_eq!(req.max_tokens, Some(64));
        assert_eq!(req.seed, Some(7));
        let tools = req.tools.unwrap();
        assert_eq!(tools[0].function.name, "calculator");
        assert_eq!(tools[0].kind, ToolKind::Function);
        assert_eq!(
            tools[0].function.parameters.as_ref().unwrap()["type"],
            "object"
        );
    }

    #[test]
    fn assistant_message_with_tool_calls_and_null_content() {
        let msg: ChatMessage = parse(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculator", "arguments": "{\"expression\":\"2+2\"}"}
            }]
        }));
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, None);
        let calls = msg.tool_calls.unwrap();
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.arguments, "{\"expression\":\"2+2\"}");
    }

    #[test]
    fn assistant_message_without_content_field_deserializes() {
        let msg: ChatMessage = parse(json!({"role": "assistant"}));
        assert_eq!(msg.content, None);
    }

    #[test]
    fn tool_message_deserializes() {
        let msg: ChatMessage =
            parse(json!({"role": "tool", "tool_call_id": "call_1", "content": "4"}));
        assert_eq!(msg, ChatMessage::tool("call_1", "4"));
    }

    #[test]
    fn stop_accepts_string_and_list() {
        let one: ChatRequest = parse(json!({"messages": [], "stop": "x"}));
        assert_eq!(one.stop.unwrap().into_vec(), vec!["x"]);
        let many: ChatRequest = parse(json!({"messages": [], "stop": ["x", "y"]}));
        assert_eq!(many.stop.unwrap().into_vec(), vec!["x", "y"]);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let req: ChatRequest = parse(json!({
            "messages": [{"role": "user", "content": "hi", "name": "bob"}],
            "logprobs": true,
            "n": 1
        }));
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn unknown_role_is_rejected() {
        let result: Result<ChatMessage, _> =
            serde_json::from_value(json!({"role": "robot", "content": "hi"}));
        assert!(result.is_err());
    }

    #[test]
    fn response_serializes_to_expected_json() {
        let response = ChatResponse {
            id: "chatcmpl-1".into(),
            object: OBJECT_CHAT_COMPLETION.into(),
            created: 1,
            model: "m".into(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage::assistant_tool_calls(
                    None,
                    vec![ToolCall::function(
                        "call_0",
                        "calculator",
                        "{\"expression\":\"1\"}",
                    )],
                ),
                finish_reason: FinishReason::ToolCalls,
            }],
            usage: Usage::new(10, 5),
        };
        assert_eq!(
            serde_json::to_value(&response).unwrap(),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": "{\"expression\":\"1\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            })
        );
    }

    #[test]
    fn text_message_omits_tool_fields() {
        assert_eq!(
            serde_json::to_value(ChatMessage::user("hi")).unwrap(),
            json!({"role": "user", "content": "hi"})
        );
    }

    #[test]
    fn stream_chunk_serializes_to_expected_json() {
        let chunk = ChatChunk {
            id: "chatcmpl-1".into(),
            object: OBJECT_CHAT_COMPLETION_CHUNK.into(),
            created: 1,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    content: Some("Hi".into()),
                    ..Delta::default()
                },
                finish_reason: None,
            }],
            usage: None,
        };
        assert_eq!(
            serde_json::to_value(&chunk).unwrap(),
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]
            })
        );
    }

    #[test]
    fn fragmented_tool_call_delta_deserializes() {
        let delta: Delta = parse(json!({
            "tool_calls": [{"index": 0, "function": {"arguments": "{\"exp"}}]
        }));
        let call = &delta.tool_calls.unwrap()[0];
        assert_eq!(call.index, Some(0));
        assert_eq!(call.id, None);
        assert_eq!(call.function.as_ref().unwrap().name, None);
        assert_eq!(
            call.function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"exp")
        );
    }

    #[test]
    fn error_body_roundtrips() {
        let body = ErrorBody::new(
            "invalid_request_error",
            "bad",
            Some("context_length_exceeded"),
        );
        let value = serde_json::to_value(&body).unwrap();
        assert_eq!(
            value,
            json!({"error": {"message": "bad", "type": "invalid_request_error", "code": "context_length_exceeded"}})
        );
        assert_eq!(parse::<ErrorBody>(value), body);
    }
}
