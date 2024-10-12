use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    style::{Style, Stylize},
    text::{Span, Text},
};

use crate::{
    error::{Error, Result},
    lm::Response,
    ollama::chat::Message,
};

pub mod state;
pub mod view;

pub mod highlight_spacing;
pub mod widget;
pub mod widget_item;
pub mod widget_rendering;
pub mod widget_state;

const TEST_OUTPUT: &str = include_str!("../../../example_output.txt");

#[derive(Clone, Debug)]
pub struct MessagesViewModel {
    /// Streaming response from the model before it's finished.
    model_stream: String,
    /// A list of [`Message`]s that constitute the history
    /// of the conversation.
    messages: VecDeque<Message>,
    state: state::MessagesState,
}

impl Default for MessagesViewModel {
    fn default() -> Self {
        Self {
            model_stream: Default::default(),
            messages: VecDeque::from([Message::Assistant(TEST_OUTPUT.into())]),
            state: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum MessagesEvent {
    Quit,
}

impl<'a> From<Message> for Text<'a> {
    fn from(value: Message) -> Self {
        let role = value.role();
        let content = value.content();

        let role = Span::styled(role, Style::default().bold());
        let content = Span::from(format!(": {content}"));

        Text::from_iter([role, content])
    }
}

impl MessagesViewModel {
    /// Get a list of text lines that are wrapped
    /// around the given `view_width`.
    fn get_message_list(&self) -> Vec<Message> {
        std::iter::once(Message::Assistant(self.model_stream.clone().into()))
            .chain(self.messages.clone())
            .collect()
    }

    pub fn handle_response(&mut self, response: Response) -> Result<()> {
        match response {
            Response::ModelInfo(_) | Response::LocalModels(_) => {
                return Err(Error::UnexpectedResponse(response))
            }
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

        Ok(())
    }

    pub fn handle_key_event(&mut self, key_event: KeyEvent) -> Option<MessagesEvent> {
        match key_event.code {
            KeyCode::Char('q') => {
                self.state.select(None);
                Some(MessagesEvent::Quit)
            }
            KeyCode::Char('j') => {
                self.state.select_next();
                None
            }
            KeyCode::Char('k') => {
                self.state.select_previous();
                None
            }
            KeyCode::Enter => {
                if self.state.selected().is_some() {
                    // TODO: enter fullscreen
                }
                None
            }
            _ => None,
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
