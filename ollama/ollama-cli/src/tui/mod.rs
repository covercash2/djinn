use std::{collections::VecDeque, sync::Arc, time::Duration};

use futures::StreamExt as _;
use input_context::{InputContext, InputMode};
use model_context::ModelContext;
use ratatui::{
    crossterm::event::Event,
    layout::{Constraint, Layout, Position},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, List, ListItem, Paragraph},
    DefaultTerminal, Frame,
};

use crate::ollama;

mod input_context;
mod model_context;

pub struct AppContext {
    input_context: InputContext,
    model_context: ModelContext,
    stream: String,
    messages: VecDeque<Arc<str>>,
}

impl AppContext {
    pub fn new(client: ollama::Client) -> Self {
        Self {
            messages: Default::default(),
            input_context: Default::default(),
            stream: Default::default(),
            model_context: ModelContext::spawn(client),
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

        let messages: Vec<ListItem> = std::iter::once(self.stream.clone().into())
            .chain(self.messages.iter().map(Clone::clone))
            .enumerate()
            .map(|(i, m)| {
                let content = Line::from(Span::raw(format!("{i}: {m}")));
                ListItem::new(content)
            })
            .collect();
        let messages = List::new(messages).block(Block::bordered().title("Messages"));
        frame.render_widget(messages, messages_area);
    }

    pub async fn run(mut self, mut terminal: DefaultTerminal) -> anyhow::Result<()> {
        let period = Duration::from_secs_f32(1.0 / 15.0);
        let mut interval = tokio::time::interval(period);
        let mut events = ratatui::crossterm::event::EventStream::new();
        loop {
            tokio::select! {
                _ = interval.tick() => { terminal.draw(|frame| self.draw(frame))?; },
                Some(Ok(Event::Key(key))) = events.next() => {
                    if let Some(app_event) = self.input_context.handle_key_event(key) {
                        match app_event {
                            AppEvent::Submit(message) => self.submit_message(message).await,
                            AppEvent::Quit => return Ok(()),
                        }
                    }
                },
                Some(response) = self.model_context.response_receiver.recv() => {
                    match response {
                        crate::lm::Response::Eos => {
                            self.messages.push_front(self.stream.clone().into());
                            self.stream.clear();
                        }
                        crate::lm::Response::Error(error_str) => {
                            if !self.stream.is_empty() {
                                self.messages.push_front(self.stream.clone().into());
                                self.stream.clear();
                            }
                            self.messages.push_front(error_str);
                        }
                        crate::lm::Response::Token(str) => {
                            self.stream.push_str(str.as_ref());
                        }
                    }
                }
            }
        }
    }

    async fn submit_message(&mut self, message: Arc<str>) {
        self.model_context
            .prompt_sender
            .send(message)
            .await
            .expect("unable to send message")
    }
}

pub enum AppEvent {
    Submit(Arc<str>),
    Quit,
}
