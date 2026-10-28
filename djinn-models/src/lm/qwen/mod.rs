//! Qwen2.5-Instruct: chat prompt format and tool-call extraction.
//!
//! The prompt format is the model's own `chat_template` (see `NOTES.md` next to this file for
//! where it comes from and how the golden fixtures were produced). It is implemented here by hand
//! rather than by a Jinja engine; `fixtures.json` pins the output to the real template.

mod pyjson;

use djinn_core::chat::{ChatMessage, Role, ToolCall, ToolSpec};
use djinn_core::values::{ToolArguments, ToolName};
use serde_json::Value;

use super::chat::{ChatTemplate, ParsedReply, next_call_id};
use pyjson::to_py_json;

const DEFAULT_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

const TOOLS_INTRO: &str = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const TOOLS_OUTRO: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// `<|im_end|>` ends a turn; `<|endoftext|>` is the other end-of-sequence id in the model's
/// generation config.
const END_OF_TURN: &[&str] = &["<|im_end|>", "<|endoftext|>"];

/// The Qwen2.5 chat format (ChatML with Hermes-style tool calls).
#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen25Template;

fn text(message: &ChatMessage) -> &str {
    message.content.as_deref().unwrap_or("")
}

fn has_tool_calls(message: &ChatMessage) -> bool {
    message.tool_calls.as_ref().is_some_and(|c| !c.is_empty())
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

impl ChatTemplate for Qwen25Template {
    fn render(&self, messages: &[ChatMessage], tools: &[ToolSpec]) -> String {
        let mut out = String::new();

        let system = messages
            .first()
            .filter(|m| m.role == Role::System)
            .map(text)
            .unwrap_or(DEFAULT_SYSTEM);
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        if tools.is_empty() {
            out.push_str("<|im_end|>\n");
        } else {
            out.push_str(TOOLS_INTRO);
            for tool in tools {
                out.push('\n');
                out.push_str(&to_py_json(tool));
            }
            out.push_str(TOOLS_OUTRO);
        }

        for (index, message) in messages.iter().enumerate() {
            match message.role {
                // A leading system message was already used as the system turn above.
                Role::System if index == 0 => {}
                Role::User | Role::System => {
                    push_turn(&mut out, role_name(message.role), text(message));
                }
                Role::Assistant if !has_tool_calls(message) => {
                    push_turn(&mut out, "assistant", text(message));
                }
                Role::Assistant => {
                    out.push_str("<|im_start|>assistant");
                    if !text(message).is_empty() {
                        out.push('\n');
                        out.push_str(text(message));
                    }
                    for call in message.tool_calls.iter().flatten() {
                        out.push_str("\n<tool_call>\n{\"name\": \"");
                        out.push_str(call.function.name.as_str());
                        out.push_str("\", \"arguments\": ");
                        out.push_str(&arguments_json(&call.function.arguments));
                        out.push_str("}\n</tool_call>");
                    }
                    out.push_str("<|im_end|>\n");
                }
                Role::Tool => {
                    let previous_is_tool = index > 0 && messages[index - 1].role == Role::Tool;
                    let next_is_tool = messages
                        .get(index + 1)
                        .is_some_and(|m| m.role == Role::Tool);
                    if !previous_is_tool {
                        out.push_str("<|im_start|>user");
                    }
                    out.push_str("\n<tool_response>\n");
                    out.push_str(text(message));
                    out.push_str("\n</tool_response>");
                    if !next_is_tool {
                        out.push_str("<|im_end|>\n");
                    }
                }
            }
        }

        out.push_str("<|im_start|>assistant\n");
        out
    }

    fn parse_output(&self, output: &str) -> ParsedReply {
        let mut calls = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();

        let mut cursor = 0;
        while let Some(found) = output[cursor..].find(TOOL_CALL_OPEN) {
            let start = cursor + found;
            let body_start = start + TOOL_CALL_OPEN.len();
            let Some(close) = output[body_start..].find(TOOL_CALL_CLOSE) else {
                break; // truncated block: everything from here on stays plain text
            };
            let body_end = body_start + close;
            match parse_call(&output[body_start..body_end]) {
                Some((name, arguments)) => {
                    calls.push(ToolCall::function(next_call_id(), name, arguments));
                    spans.push((start, body_end + TOOL_CALL_CLOSE.len()));
                    cursor = body_end + TOOL_CALL_CLOSE.len();
                }
                // Invalid JSON or shape: leave it as text and keep looking after its opening tag.
                None => cursor = body_start,
            }
        }

        if calls.is_empty() {
            return ParsedReply {
                content: Some(output.to_owned()),
                tool_calls: calls,
            };
        }

        let mut remaining = String::new();
        let mut position = 0;
        for (start, end) in spans {
            remaining.push_str(&output[position..start]);
            position = end;
        }
        remaining.push_str(&output[position..]);
        let remaining = remaining.trim();
        ParsedReply {
            content: (!remaining.is_empty()).then(|| remaining.to_owned()),
            tool_calls: calls,
        }
    }

    fn tool_call_start(&self) -> Option<&'static str> {
        Some(TOOL_CALL_OPEN)
    }

    fn end_of_turn_markers(&self) -> &'static [&'static str] {
        END_OF_TURN
    }
}

fn push_turn(out: &mut String, role: &str, content: &str) {
    out.push_str("<|im_start|>");
    out.push_str(role);
    out.push('\n');
    out.push_str(content);
    out.push_str("<|im_end|>\n");
}

/// The template renders arguments as a JSON object; the wire format carries them as a string.
fn arguments_json(arguments: &ToolArguments) -> String {
    to_py_json(&arguments.to_value())
}

/// Parses the body of one `<tool_call>` block: a JSON object with a string `name` and an
/// `arguments` object. A missing `arguments` means no arguments; arguments sent as a JSON-encoded
/// string are accepted if they decode to an object.
fn parse_call(body: &str) -> Option<(ToolName, ToolArguments)> {
    let Value::Object(mut object) = serde_json::from_str::<Value>(body.trim()).ok()? else {
        return None;
    };
    let Value::String(name) = object.remove("name")? else {
        return None;
    };
    // A name that is not a valid tool name cannot be called, so the block stays plain text.
    let name = ToolName::new(name).ok()?;
    let arguments = match object.remove("arguments") {
        None | Some(Value::Null) => ToolArguments::empty(),
        Some(Value::Object(object)) => ToolArguments::from_object(object),
        Some(Value::String(encoded)) => ToolArguments::new(encoded).ok()?,
        Some(_) => return None,
    };
    Some((name, arguments))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        name: String,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolSpec>,
        expected: String,
    }

    fn fixtures() -> Vec<Fixture> {
        serde_json::from_str(include_str!("fixtures.json")).expect("fixtures.json is valid")
    }

    /// Every fixture was produced by rendering the model's real Jinja template; see NOTES.md.
    #[test]
    fn render_matches_the_official_template_for_every_fixture() {
        let fixtures = fixtures();
        assert!(fixtures.len() >= 10, "fixtures went missing");
        for fixture in fixtures {
            let rendered = Qwen25Template.render(&fixture.messages, &fixture.tools);
            assert_eq!(
                rendered, fixture.expected,
                "fixture `{}` renders differently from the official template",
                fixture.name
            );
        }
    }

    #[test]
    fn plain_conversation_has_default_system_and_open_assistant_turn() {
        let rendered = Qwen25Template.render(&[ChatMessage::user("hi")], &[]);
        assert_eq!(
            rendered,
            format!(
                "<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
            )
        );
    }

    #[test]
    fn tool_call_arguments_render_as_a_json_object() {
        let messages = [
            ChatMessage::user("x"),
            ChatMessage::assistant_tool_calls(
                None,
                vec![ToolCall::from_static("c", "f", r#"{"a":1}"#)],
            ),
        ];
        let rendered = Qwen25Template.render(&messages, &[]);
        assert!(rendered.contains("{\"name\": \"f\", \"arguments\": {\"a\": 1}}"));
    }

    #[test]
    fn a_call_with_an_invalid_tool_name_stays_plain_text() {
        let text = r#"<tool_call>{"name":"do it!","arguments":{}}</tool_call>"#;
        let reply = Qwen25Template.parse_output(text);
        assert!(reply.tool_calls.is_empty());
        assert_eq!(reply.content.as_deref(), Some(text));
    }

    #[test]
    fn empty_tool_calls_list_renders_as_plain_assistant_message() {
        let mut message = ChatMessage::assistant("hello");
        message.tool_calls = Some(vec![]);
        let rendered = Qwen25Template.render(&[ChatMessage::user("x"), message], &[]);
        assert!(rendered.contains("<|im_start|>assistant\nhello<|im_end|>\n"));
    }

    fn calls(reply: &ParsedReply) -> Vec<(&str, &str)> {
        reply
            .tool_calls
            .iter()
            .map(|c| (c.function.name.as_str(), c.function.arguments.as_str()))
            .collect()
    }

    #[test]
    fn parses_one_call() {
        let reply = Qwen25Template.parse_output(
            r#"<tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}</tool_call>"#,
        );
        assert_eq!(
            calls(&reply),
            vec![("calculator", r#"{"expression":"2+2"}"#)]
        );
        assert_eq!(reply.content, None);
        assert_eq!(
            reply.tool_calls[0].kind,
            djinn_core::chat::ToolKind::Function
        );
    }

    #[test]
    fn parses_two_calls_in_order_with_distinct_ids() {
        let reply = Qwen25Template.parse_output(
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {\"x\": 1}}\n</tool_call>",
        );
        assert_eq!(calls(&reply), vec![("a", "{}"), ("b", r#"{"x":1}"#)]);
        assert_ne!(reply.tool_calls[0].id, reply.tool_calls[1].id);
        assert_eq!(reply.content, None);
    }

    #[test]
    fn ids_are_unique_across_replies() {
        let text = r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#;
        let first = Qwen25Template.parse_output(text);
        let second = Qwen25Template.parse_output(text);
        assert_ne!(first.tool_calls[0].id, second.tool_calls[0].id);
    }

    #[test]
    fn preamble_is_kept_as_content() {
        let reply = Qwen25Template.parse_output(
            "Let me check.\n<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"1\"}}\n</tool_call>",
        );
        assert_eq!(reply.content.as_deref(), Some("Let me check."));
        assert_eq!(reply.tool_calls.len(), 1);
    }

    #[test]
    fn text_after_a_call_is_kept_too() {
        let reply = Qwen25Template
            .parse_output("before <tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call> after");
        assert_eq!(reply.content.as_deref(), Some("before  after"));
    }

    #[test]
    fn truncated_call_is_plain_text() {
        let text = "Working on it <tool_call>\n{\"name\": \"calc";
        let reply = Qwen25Template.parse_output(text);
        assert!(reply.tool_calls.is_empty());
        assert_eq!(reply.content.as_deref(), Some(text));
    }

    #[test]
    fn call_missing_its_closing_tag_is_plain_text() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{}}";
        let reply = Qwen25Template.parse_output(text);
        assert!(reply.tool_calls.is_empty());
        assert_eq!(reply.content.as_deref(), Some(text));
    }

    #[test]
    fn invalid_json_is_plain_text() {
        let text = "<tool_call>{name: calculator}</tool_call>";
        let reply = Qwen25Template.parse_output(text);
        assert!(reply.tool_calls.is_empty());
        assert_eq!(reply.content.as_deref(), Some(text));
    }

    #[test]
    fn invalid_block_does_not_hide_a_later_valid_one() {
        let reply = Qwen25Template.parse_output(
            "<tool_call>oops</tool_call> then <tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>",
        );
        assert_eq!(calls(&reply), vec![("a", "{}")]);
        assert_eq!(
            reply.content.as_deref(),
            Some("<tool_call>oops</tool_call> then")
        );
    }

    #[test]
    fn missing_name_is_not_a_call() {
        let reply = Qwen25Template.parse_output(r#"<tool_call>{"arguments":{}}</tool_call>"#);
        assert!(reply.tool_calls.is_empty());
    }

    #[test]
    fn non_string_or_empty_name_is_not_a_call() {
        assert!(
            Qwen25Template
                .parse_output(r#"<tool_call>{"name":1,"arguments":{}}</tool_call>"#)
                .tool_calls
                .is_empty()
        );
        assert!(
            Qwen25Template
                .parse_output(r#"<tool_call>{"name":"","arguments":{}}</tool_call>"#)
                .tool_calls
                .is_empty()
        );
    }

    #[test]
    fn missing_arguments_means_no_arguments() {
        let reply =
            Qwen25Template.parse_output(r#"<tool_call>{"name":"current_time"}</tool_call>"#);
        assert_eq!(calls(&reply), vec![("current_time", "{}")]);
    }

    #[test]
    fn string_encoded_arguments_are_accepted_when_they_decode_to_an_object() {
        let reply = Qwen25Template
            .parse_output(r#"<tool_call>{"name":"a","arguments":"{\"x\":1}"}</tool_call>"#);
        assert_eq!(calls(&reply), vec![("a", r#"{"x":1}"#)]);

        let bad =
            Qwen25Template.parse_output(r#"<tool_call>{"name":"a","arguments":"[1]"}</tool_call>"#);
        assert!(bad.tool_calls.is_empty());
    }

    #[test]
    fn non_object_arguments_are_rejected() {
        let reply =
            Qwen25Template.parse_output(r#"<tool_call>{"name":"a","arguments":[1]}</tool_call>"#);
        assert!(reply.tool_calls.is_empty());
    }

    #[test]
    fn plain_text_is_returned_untouched() {
        let reply = Qwen25Template.parse_output("  The answer is 4.\n");
        assert!(reply.tool_calls.is_empty());
        assert_eq!(reply.content.as_deref(), Some("  The answer is 4.\n"));
    }

    #[test]
    fn empty_output_is_empty_content() {
        let reply = Qwen25Template.parse_output("");
        assert!(reply.tool_calls.is_empty());
        assert_eq!(reply.content.as_deref(), Some(""));
    }

    #[test]
    fn exposes_markers_for_streaming() {
        assert_eq!(Qwen25Template.tool_call_start(), Some("<tool_call>"));
        assert!(Qwen25Template.end_of_turn_markers().contains(&"<|im_end|>"));
    }
}
