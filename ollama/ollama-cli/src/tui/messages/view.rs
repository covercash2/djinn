use std::{borrow::Cow, sync::Arc};

use ratatui::{
    layout::Rect,
    style::{Color, Style, Stylize as _},
    text::{Line, Span, Text},
    widgets::{Block, List, ListItem},
    Frame,
};

use crate::ollama::chat::Message;

use super::MessagesViewModel;

const ELIPSIS: &str = "[...]";

#[derive(Default)]
struct MessageViewBuilder {
    max_height: u16,
    message_cell_width: u16,
    remaining_lines: u16,
    consumed_lines: u16,
}

#[derive(Debug, Clone)]
pub struct MessageContent {
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
        self.render_stateful_widget(table, parent, &mut view_model.state.list_state);
    }
}
