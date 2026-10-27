//! Validation of chat requests that deserialization alone cannot express.

use std::collections::HashSet;

use thiserror::Error;

use super::{ChatRequest, Role};

/// The most stop strings a request may carry.
pub const MAX_STOP_SEQUENCES: usize = 4;

/// Why a [`ChatRequest`] was rejected.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ChatRequestError {
    #[error("`messages` must contain at least one message")]
    EmptyMessages,
    #[error("`stop` may contain at most {MAX_STOP_SEQUENCES} sequences, got {0}")]
    TooManyStopSequences(usize),
    #[error("message {index}: `tool` messages must have a `tool_call_id`")]
    MissingToolCallId { index: usize },
    #[error(
        "message {index}: tool_call_id `{id}` does not match a tool call made by an earlier assistant message"
    )]
    UnknownToolCallId { index: usize, id: String },
}

/// Checks the rules from the chat-completions spec: a non-empty conversation, at most
/// [`MAX_STOP_SEQUENCES`] stop strings, and tool results that answer an earlier tool call.
pub fn validate_request(request: &ChatRequest) -> Result<(), ChatRequestError> {
    if request.messages.is_empty() {
        return Err(ChatRequestError::EmptyMessages);
    }

    if let Some(stop) = &request.stop {
        let count = stop.to_vec().len();
        if count > MAX_STOP_SEQUENCES {
            return Err(ChatRequestError::TooManyStopSequences(count));
        }
    }

    let mut issued: HashSet<&str> = HashSet::new();
    for (index, message) in request.messages.iter().enumerate() {
        match message.role {
            Role::Assistant => {
                for call in message.tool_calls.iter().flatten() {
                    issued.insert(call.id.as_str());
                }
            }
            Role::Tool => {
                let Some(id) = message.tool_call_id.as_deref() else {
                    return Err(ChatRequestError::MissingToolCallId { index });
                };
                if !issued.contains(id) {
                    return Err(ChatRequestError::UnknownToolCallId {
                        index,
                        id: id.to_owned(),
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
                vec![ToolCall::function("call_1", "calculator", "{}")],
            ),
            ChatMessage::tool("call_1", "4"),
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
    fn four_stop_sequences_are_allowed_five_are_not() {
        let mut req = request(vec![ChatMessage::user("hi")]);
        req.stop = Some(StopSpec::Many(vec![
            "a".into(),
            "b".into(),
            "c".into(),
            "d".into(),
        ]));
        assert_eq!(validate_request(&req), Ok(()));

        req.stop = Some(StopSpec::Many(vec![
            "a".into(),
            "b".into(),
            "c".into(),
            "d".into(),
            "e".into(),
        ]));
        assert_eq!(
            validate_request(&req),
            Err(ChatRequestError::TooManyStopSequences(5))
        );
    }

    #[test]
    fn single_stop_string_is_allowed() {
        let mut req = request(vec![ChatMessage::user("hi")]);
        req.stop = Some(StopSpec::One("x".into()));
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
            ChatMessage::tool("call_9", "4"),
        ]);
        assert_eq!(
            validate_request(&req),
            Err(ChatRequestError::UnknownToolCallId {
                index: 1,
                id: "call_9".into()
            })
        );
    }

    #[test]
    fn tool_result_with_mismatched_id_is_rejected() {
        let mut messages = call_round_trip();
        messages[2] = ChatMessage::tool("call_other", "4");
        assert_eq!(
            validate_request(&request(messages)),
            Err(ChatRequestError::UnknownToolCallId {
                index: 2,
                id: "call_other".into()
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
        let mut tool = ChatMessage::tool("x", "4");
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
        assert!(ChatRequestError::TooManyStopSequences(5)
            .to_string()
            .contains("at most 4"));
    }
}
