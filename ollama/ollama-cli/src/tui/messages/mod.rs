use std::collections::VecDeque;

use ratatui::{
    layout::Rect,
    text::Text,
    widgets::{Block, List, ListItem},
    Frame,
};

use crate::{lm::Response, ollama::chat::Message};

#[derive(Default, Clone, Debug)]
pub struct MessagesViewModel {
    /// Streaming response from the model before it's finished.
    model_stream: String,
    /// A list of [`Message`]s that constitute the history
    /// of the conversation.
    messages: VecDeque<Message>,
}

impl MessagesViewModel {
    /// Get a list of text lines that are wrapped
    /// around the given `view_width`.
    fn get_message_list(&self) -> Vec<String> {
        std::iter::once(self.model_stream.clone())
            .chain(self.messages.iter().map(|message| message.to_string()))
            .collect()
    }

    pub fn handle_response(&mut self, response: Response) {
        match response {
            Response::Eos => {
                let message = Message::Assistant(self.model_stream.clone().into());
                self.push_message(message);
                self.clear_stream();
            }
            Response::Error(error_str) => {
                if !self.is_stream_empty() {
                    let message = Message::Assistant(self.model_stream.clone().into());
                    self.push_message(message);
                    self.clear_stream();
                }
                // TODO: make an error state
                let message = Message::User(error_str);
                self.push_message(message);
            }
            Response::Token(str) => {
                self.model_stream.push_str(str.as_ref());
            }
        }
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push_front(message);
    }

    pub fn history(&self) -> Vec<Message> {
        self.messages.clone().into()
    }

    fn is_stream_empty(&self) -> bool {
        self.model_stream.is_empty()
    }

    fn clear_stream(&mut self) {
        self.model_stream.clear();
    }
}

#[extend::ext(name = MessagesView)]
pub impl<'a> Frame<'a> {
    fn message_view(&mut self, parent: Rect, view_model: &MessagesViewModel) {
        let max_message_width = parent.width - 2;
        let wrap = textwrap::Options::new(max_message_width.into());

        let messages = view_model.get_message_list();

        let messages: Vec<ListItem> = messages
            .iter()
            .flat_map(|message| {
                textwrap::wrap(message, wrap.clone())
                    .into_iter()
                    .map(|line| ListItem::from(Text::from(line)))
                    .collect::<Vec<_>>()
            })
            .collect();
        let messages = List::new(messages).block(Block::bordered().title("Messages"));
        self.render_widget(messages, parent);
    }
}
