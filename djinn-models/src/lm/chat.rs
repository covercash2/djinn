//! Chat prompt rendering and model-output parsing, independent of any particular model.

use djinn_core::chat::{ChatMessage, FinishReason, ToolCall, ToolSpec, Usage};
use djinn_core::values::{MaxTokens, StopSequence, Temperature, ToolCallId, TopP};

/// What a model said, split into text and tool calls.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedReply {
    /// The reply text. `None` when the model only made tool calls.
    pub content: Option<String>,
    /// Complete, valid tool calls in the order the model made them.
    pub tool_calls: Vec<ToolCall>,
}

/// Turns conversations into the prompt format a model was trained on, and model output back into
/// text plus tool calls.
pub trait ChatTemplate: Send + Sync {
    /// Renders the conversation, ending with an opened assistant turn ready for generation.
    fn render(&self, messages: &[ChatMessage], tools: &[ToolSpec]) -> String;

    /// Splits generated text into content and tool calls. Markup that is incomplete or invalid is
    /// kept as plain text.
    fn parse_output(&self, text: &str) -> ParsedReply;

    /// The literal text that begins a tool call in generated output, if the format has one. Used
    /// to avoid streaming tool-call markup to the client as ordinary text.
    fn tool_call_start(&self) -> Option<&'static str>;

    /// Strings that mean the model has finished its turn, besides its end-of-sequence token. The
    /// generator stops on them and never shows them.
    fn end_of_turn_markers(&self) -> &'static [&'static str];
}

/// Returns a call id that is unique within this process.
pub(crate) fn next_call_id() -> ToolCallId {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    ToolCallId::new(format!("call_{}", NEXT.fetch_add(1, Ordering::Relaxed)))
        .expect("generated ids are valid")
}

/// Per-request generation settings. `None` means "use the server default".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatParams {
    pub max_tokens: Option<MaxTokens>,
    pub temperature: Option<Temperature>,
    pub top_p: Option<TopP>,
    pub seed: Option<u64>,
    /// Generation ends before any of these strings would appear in the output.
    pub stop: Vec<StopSequence>,
}

/// Everything needed to generate one assistant reply.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatInput {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolSpec>,
    pub params: ChatParams,
}

/// The final result of a chat generation.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatOutcome {
    /// The complete reply text with stop strings trimmed and tool-call markup removed; `None`
    /// when the model only made tool calls.
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish: FinishReason,
    pub usage: Usage,
}

/// What a chat generation produces: text that is safe to show as it arrives, then the outcome.
///
/// Tool-call markup and partial stop strings are never part of a [`ChatEvent::Text`]. If the
/// model starts a tool call that turns out to be invalid, the held-back markup is released as a
/// final `Text` event so that the streamed text always equals the outcome's content.
#[derive(Clone, Debug, PartialEq)]
pub enum ChatEvent {
    Text(String),
    Done(ChatOutcome),
}

/// A stream of [`ChatEvent`]s borrowed from the model that produces it.
pub type ChatStream<'a> =
    std::pin::Pin<Box<dyn futures::Stream<Item = crate::Result<ChatEvent>> + Send + 'a>>;
