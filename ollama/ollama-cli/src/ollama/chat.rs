use std::sync::Arc;

use ollama_rs::generation::chat::{
    request::ChatMessageRequest, ChatMessage, ChatMessageResponseStream,
};
use serde::{Deserialize, Serialize};
use strum::{EnumDiscriminants, EnumString};

use super::{Client, ModelName};

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub prompt: Arc<str>,
    pub model: ModelName,
    pub history: Vec<Message>,
}

#[derive(Debug, Clone, strum::Display, EnumDiscriminants, Serialize, Deserialize)]
#[strum_discriminants(name(MessageRole))]
#[strum_discriminants(derive(EnumString))]
#[strum_discriminants(strum(serialize_all = "lowercase"))]
pub enum Message {
    #[strum(serialize = "assistant: {0}")]
    Assistant(Arc<str>),
    #[strum(serialize = "user: {0}")]
    User(Arc<str>),
    #[strum(serialize = "system: {0}")]
    System(Arc<str>),
}

impl Message {
    pub fn role(&self) -> &'static str {
        match self {
            Message::Assistant(_) => "assistant",
            Message::User(_) => "user",
            Message::System(_) => "system",
        }
    }

    pub fn content(&self) -> Arc<str> {
        match self {
            Message::Assistant(arc) | Message::User(arc) | Message::System(arc) => arc.clone(),
        }
    }
}

impl<'a> From<(MessageRole, &'a str)> for Message {
    fn from(value: (MessageRole, &'a str)) -> Self {
        let (role, message) = value;
        let message: Arc<str> = message.into();
        match role {
            MessageRole::Assistant => Message::Assistant(message),
            MessageRole::User => Message::User(message),
            MessageRole::System => Message::System(message),
        }
    }
}

impl From<ChatRequest> for ChatMessageRequest {
    fn from(value: ChatRequest) -> Self {
        let ChatRequest {
            prompt,
            model,
            history,
        } = value;

        let messages: Vec<ChatMessage> = history
            .into_iter()
            .map(|message| match message {
                Message::User(msg) => ChatMessage::user(msg.to_string()),
                Message::System(msg) => ChatMessage::system(msg.to_string()),
                Message::Assistant(msg) => ChatMessage::assistant(msg.to_string()),
            })
            .chain(std::iter::once(ChatMessage::user(prompt.to_string())))
            .collect();

        ChatMessageRequest::new(model.to_string(), messages)
    }
}

impl Client {
    pub async fn chat(
        &self,
        request: ChatRequest,
    ) -> ollama_rs::error::Result<ChatMessageResponseStream> {
        let request = request.into();
        self.client.send_chat_messages_stream(request).await
    }
}
