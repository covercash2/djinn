use std::sync::Arc;

use ollama_rs::generation::chat::{
    request::ChatMessageRequest, ChatMessage, ChatMessageResponseStream,
};

use super::{Client, ModelName};

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub prompt: Arc<str>,
    pub model: ModelName,
    pub history: Vec<Message>,
}

#[derive(Debug, Clone, strum::Display)]
pub enum Message {
    #[strum(serialize = "assistant: {0}")]
    Assistant(Arc<str>),
    #[strum(serialize = "user: {0}")]
    User(Arc<str>),
    #[strum(serialize = "system: {0}")]
    System(Arc<str>),
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
