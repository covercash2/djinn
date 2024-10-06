use std::{borrow::Cow, collections::VecDeque, sync::Arc};

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, Cell, List, ListItem, ListState, Row, Table, TableState},
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
    state: ListState,
}

impl Default for MessagesViewModel {
    fn default() -> Self {
        Self {
            model_stream: Default::default(),
            messages: VecDeque::from([Message::Assistant(TEST_OUTPUT.into())]),
            state: ListState::default(),
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

#[derive(Default)]
struct MessageViewBuilder {
    max_height: u16,
    message_cell_width: u16,
    remaining_lines: u16,
    consumed_lines: u16,
}

struct MessageContent {
    role: &'static str,
    content: Vec<Arc<str>>,
    height: u16,
}

impl MessageContent {
    fn empty() -> Self {
        MessageContent {
            role: "empty",
            content: Default::default(),
            height: 0,
        }
    }
}

impl MessageViewBuilder {
    fn new(max_height: u16, width: u16) -> Self {
        MessageViewBuilder {
            max_height,
            message_cell_width: width,
            remaining_lines: max_height,
            ..Default::default()
        }
    }

    fn make_row(&mut self, message: &Message) -> MessageContent {
        if self.remaining_lines > 0 {
            let content = self.make_message_content(message);

            let consumed_lines = self.consumed_lines + content.height;
            debug_assert!(self.remaining_lines >= content.height);
            let remaining_lines = self.remaining_lines - content.height;

            debug_assert_eq!(self.max_height, consumed_lines + remaining_lines);

            self.consumed_lines = consumed_lines;
            self.remaining_lines = remaining_lines;

            content
        } else {
            MessageContent::empty()
        }
    }

    fn make_message_content(&self, message: &Message) -> MessageContent {
        let role = message.role();
        let content = message.content();
        tracing::info!(role, %content, "creating message row");

        let content_lines = fit_content(&content, self.message_cell_width, self.remaining_lines);

        let height = (content_lines.len())
            .try_into()
            .expect("should be able to coerce this usize into a u16");

        let content: Vec<Arc<str>> = content_lines
            .into_iter()
            .map(|line: Cow<'_, str>| line.into())
            .collect();

        MessageContent {
            role,
            content,
            height,
        }
    }
}

fn fit_content(content: &str, width: u16, height: u16) -> Vec<Cow<'_, str>> {
    let mut content_lines: Vec<Cow<'_, str>> = textwrap::wrap(content, usize::from(width))
        .into_iter()
        .take(height.into())
        .collect();

    if (height as usize) <= content_lines.len() {
        content_lines.pop();
        content_lines.push(ELIPSIS.into());
    }

    content_lines
}

fn fit_messages(messages: &[Message], max_height: u16, message_cell_width: u16) -> Vec<ListItem> {
    messages
        .iter()
        .scan(
            MessageViewBuilder::new(max_height, message_cell_width),
            move |builder, message| Some(builder.make_row(message)),
        )
        .map(|content| {
            let MessageContent {
                role,
                content,
                height: _,
            } = content;

            let mut content = content.into_iter();
            let first = Line::from_iter([
                Span::from(role).bold(),
                Span::from(": "),
                Span::from(content.next().unwrap_or_default().to_string()),
            ]);

            let rest = content.map(|line: Arc<str>| Line::from(line.to_string()));

            let lines = std::iter::once(first).chain(rest);

            ListItem::from(Text::from_iter(lines))
        })
        .collect()
}

#[extend::ext(name = MessagesView)]
pub impl<'a> Frame<'a> {
    fn messages_view(&mut self, parent: Rect, style: Style, view_model: &mut MessagesViewModel) {
        let messages = view_model.get_message_list();

        let role_cell_width = 10;
        let max_height = parent.height - 2;
        let message_cell_width = parent.width - 3 - role_cell_width;

        let messages = fit_messages(&messages, max_height, message_cell_width);

        tracing::info!(rows.len = messages.len());

        // let widths = [Constraint::Length(10), Constraint::Max(message_cell_width)];

        let table = List::new(messages)
            .block(Block::bordered().style(style))
            .highlight_style(
                Style::default()
                    .bg(style.fg.unwrap_or(Color::Cyan))
                    .fg(style.bg.unwrap_or(Color::Black)),
            );
        self.render_stateful_widget(table, parent, &mut view_model.state);
    }
}
