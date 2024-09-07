use std::sync::Arc;

use input_context::{InputContext, InputMode};
use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    layout::{Constraint, Layout, Position},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, List, ListItem, Paragraph},
    DefaultTerminal, Frame,
};

use crate::ollama;

mod input_context;

pub struct AppContext {
    client: ollama::Client,
    input_context: InputContext,
    messages: Vec<Arc<str>>,
}

impl AppContext {
    pub fn new(client: ollama::Client) -> Self {
        Self {
            client,
            messages: Vec::new(),
            input_context: Default::default(),
        }
    }

    fn draw(&self, frame: &mut Frame) {
        let vertical = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Min(1),
        ]);

        let [help_area, input_area, messages_area] = vertical.areas(frame.area());

        let (msg, style) = match self.input_context.mode {
            InputMode::Normal => (
                vec![
                    "Press ".into(),
                    "q".bold(),
                    " to exit, ".into(),
                    "e".bold(),
                    " to start editing.".bold(),
                ],
                Style::default().add_modifier(Modifier::RAPID_BLINK),
            ),
            InputMode::Edit => (
                vec![
                    "Press ".into(),
                    "Esc".bold(),
                    " to stop editing, ".into(),
                    "Enter".bold(),
                    " to record the message".into(),
                ],
                Style::default(),
            ),
        };

        let text = Text::from(Line::from(msg)).patch_style(style);
        let help_message = Paragraph::new(text);
        frame.render_widget(help_message, help_area);

        let input = Paragraph::new(self.input_context.input.as_str())
            .style(match self.input_context.mode {
                InputMode::Normal => Style::default(),
                InputMode::Edit => Style::default().fg(Color::Yellow),
            })
            .block(Block::bordered().title("Input"));
        frame.render_widget(input, input_area);

        match self.input_context.mode {
            InputMode::Normal => {}
            InputMode::Edit => frame.set_cursor_position(Position::new(
                input_area.x + self.input_context.cursor_position as u16 + 1,
                input_area.y + 1,
            )),
        }

        let messages: Vec<ListItem> = self
            .messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let content = Line::from(Span::raw(format!("{i}: {m}")));
                ListItem::new(content)
            })
            .collect();
        let messages = List::new(messages).block(Block::bordered().title("Messages"));
        frame.render_widget(messages, messages_area);
    }

    pub fn run(mut self, mut terminal: DefaultTerminal) -> anyhow::Result<()> {
        loop {
            terminal.draw(|frame| self.draw(frame))?;

            if let Event::Key(key) = event::read()? {
                if let Some(app_event) = self.input_context.handle_key_event(key) {
                    match app_event {
                        AppEvent::Submit(message) => self.submit_message(message),
                        AppEvent::Quit => return Ok(()),
                    }
                }
            }
        }
    }

    fn submit_message(&mut self, message: Arc<str>) {
        self.messages.push(message);
    }
}

pub enum AppEvent {
    Submit(Arc<str>),
    Quit,
}
