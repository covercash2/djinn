use std::collections::VecDeque;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Style, Stylize},
    text::{Span, Text},
    widgets::{Block, Cell, Row, Table, TableState},
    Frame,
};

use crate::{lm::Response, ollama::chat::Message};

const TEST_OUTPUT: &str = include_str!("../../../example_output.txt");
const ELIPSIS: &str = "[...]";

#[derive(Clone, Debug)]
pub struct MessagesViewModel {
    /// Streaming response from the model before it's finished.
    model_stream: String,
    /// A list of [`Message`]s that constitute the history
    /// of the conversation.
    messages: VecDeque<Message>,
    state: TableState,
}

impl Default for MessagesViewModel {
    fn default() -> Self {
        Self {
            model_stream: Default::default(),
            messages: VecDeque::from([Message::Assistant(TEST_OUTPUT.into())]),
            state: TableState::default(),
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

    pub fn handle_key_event(&mut self, key_event: KeyEvent) -> Option<MessagesEvent> {
        match key_event.code {
            KeyCode::Char('q') => {
                self.state.select(None);
                Some(MessagesEvent::Quit)
            },
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

#[extend::ext(name = MessagesView)]
pub impl<'a> Frame<'a> {
    fn messages_view(&mut self, parent: Rect, style: Style, view_model: &mut MessagesViewModel) {
        let messages = view_model.get_message_list();

        let role_cell_width = 10;
        let max_height = parent.height - 3;
        let message_cell_width = parent.width - 3 - role_cell_width;

        let messages = messages.iter().map(|message| {
            let role = message.role();
            let content = message.content().clone();
            tracing::info!(role, %content, "creating message row");
            let role = Cell::from(Span::from(role).bold());

            let content_lines = textwrap::wrap(&content, usize::from(message_cell_width));
            let num_lines: u16 = content_lines
                .iter()
                .take(max_height.into())
                .count()
                .try_into()
                .expect("unexpected overflow trying to coerce usize into u16");

            let mut content: String = content_lines.iter().take((max_height - 1).into()).fold(
                String::new(),
                |mut acc, line| {
                    acc.push_str(line);
                    acc.push_str(" \n");
                    acc
                },
            );

            if (max_height as usize) <= content_lines.len() {
                content.push_str(ELIPSIS);
            }

            let content_cell = Cell::from(content);

            Row::from_iter([role, content_cell]).height(num_lines)
        });

        let widths = [Constraint::Length(10), Constraint::Max(message_cell_width)];

        let table = Table::new(messages, widths)
            .block(Block::bordered().style(style))
            .highlight_style(
                Style::default()
                    .bg(style.fg.unwrap_or(Color::Cyan))
                    .fg(style.bg.unwrap_or(Color::Black)),
            );
        self.render_stateful_widget(table, parent, &mut view_model.state);
    }
}
