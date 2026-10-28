//! Validation of chat requests that deserialization alone cannot express.

use std::collections::HashSet;

use thiserror::Error;

use super::{ChatRequest, Role};
use crate::values::ToolCallId;

/// Why a [`ChatRequest`] was rejected.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ChatRequestError {
    #[error("`messages` must contain at least one message")]
    EmptyMessages,
    #[error("message {index}: `tool` messages must have a `tool_call_id`")]
    MissingToolCallId { index: usize },
    #[error(
        "message {index}: tool_call_id `{id}` does not match a tool call made by an earlier assistant message"
    )]
    UnknownToolCallId { index: usize, id: ToolCallId },
}

/// Checks the rules that no single value can check for itself: a non-empty conversation, and tool
/// results that answer an earlier tool call. (Stop strings, tool names and the rest are checked
/// when they are deserialized; see [`crate::values`].)
pub fn validate_request(request: &ChatRequest) -> Result<(), ChatRequestError> {
    if request.messages.is_empty() {
        return Err(ChatRequestError::EmptyMessages);
    }

    let mut issued: HashSet<&ToolCallId> = HashSet::new();
    for (index, message) in request.messages.iter().enumerate() {
        match message.role {
            Role::Assistant => {
                for call in message.tool_calls.iter().flatten() {
                    issued.insert(&call.id);
                }
            }
            Role::Tool => {
                let Some(id) = message.tool_call_id.as_ref() else {
                    return Err(ChatRequestError::MissingToolCallId { index });
                };
                if !issued.contains(id) {
                    return Err(ChatRequestError::UnknownToolCallId {
                        index,
                        id: id.clone(),
                    });
                }
            }
            Role::System | Role::User => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{ChatMessage, StopSpec, ToolCall};
    use super::*;
    use crate::values::StopSequence;

    fn request(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            messages,
            ..ChatRequest::default()
        }
    }

    fn call_round_trip() -> Vec<ChatMessage> {
        vec![
            ChatMessage::user("2+2?"),
            ChatMessage::assistant_tool_calls(
                None,
                vec![ToolCall::from_static("call_1", "calculator", "{}")],
            ),
            ChatMessage::tool(ToolCallId::from_static("call_1"), "4"),
        ]
    }

    #[test]
    fn minimal_request_is_valid() {
        assert_eq!(
            validate_request(&request(vec![ChatMessage::user("hi")])),
            Ok(())
        );
    }

    #[test]
    fn empty_messages_are_rejected() {
        assert_eq!(
            validate_request(&request(vec![])),
            Err(ChatRequestError::EmptyMessages)
        );
    }

    #[test]
    fn stop_strings_are_checked_by_their_types_not_here() {
        let mut req = request(vec![ChatMessage::user("hi")]);
        req.stop = Some(StopSpec::One(StopSequence::from_static("x")));
        assert_eq!(validate_request(&req), Ok(()));
    }

    #[test]
    fn tool_result_after_matching_call_is_valid() {
        assert_eq!(validate_request(&request(call_round_trip())), Ok(()));
    }

    #[test]
    fn tool_result_without_any_call_is_rejected() {
        let req = request(vec![
            ChatMessage::user("hi"),
            ChatMessage::tool(ToolCallId::from_static("call_9"), "4"),
        ]);
        assert_eq!(
            validate_request(&req),
            Err(ChatRequestError::UnknownToolCallId {
                index: 1,
                id: ToolCallId::from_static("call_9")
            })
        );
    }

    #[test]
    fn tool_result_with_mismatched_id_is_rejected() {
        let mut messages = call_round_trip();
        messages[2] = ChatMessage::tool(ToolCallId::from_static("call_other"), "4");
        assert_eq!(
            validate_request(&request(messages)),
            Err(ChatRequestError::UnknownToolCallId {
                index: 2,
                id: ToolCallId::from_static("call_other")
            })
        );
    }

    #[test]
    fn tool_result_before_its_call_is_rejected() {
        let mut messages = call_round_trip();
        messages.swap(1, 2);
        assert!(matches!(
            validate_request(&request(messages)),
            Err(ChatRequestError::UnknownToolCallId { index: 1, .. })
        ));
    }

    #[test]
    fn tool_message_without_id_is_rejected() {
        let mut tool = ChatMessage::tool(ToolCallId::from_static("x"), "4");
        tool.tool_call_id = None;
        let req = request(vec![ChatMessage::user("hi"), tool]);
        assert_eq!(
            validate_request(&req),
            Err(ChatRequestError::MissingToolCallId { index: 1 })
        );
    }

    #[test]
    fn errors_have_readable_messages() {
        assert!(ChatRequestError::EmptyMessages
            .to_string()
            .contains("at least one message"));
    }
}
