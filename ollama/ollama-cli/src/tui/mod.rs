use std::{collections::VecDeque, sync::Arc, time::Duration};

use extend::ext;
use futures::StreamExt as _;
use input_context::{InputContext, InputMode};
use model_context::ModelContext;
use ratatui::{
    crossterm::event::Event,
    layout::{Constraint, Layout, Position, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Text},
    widgets::{Block, List, ListItem, Paragraph},
    DefaultTerminal, Frame,
};

use crate::{
    lm::{Prompt, Response},
    ollama::{
        self,
        chat::{ChatRequest, Message},
    },
};

mod input_context;
mod model_context;

pub struct AppContext {
    input_context: InputContext,
    model_context: ModelContext,
    messages_view_model: MessagesViewModel,
}

impl AppContext {
    pub fn new(client: ollama::Client) -> Self {
        Self {
            input_context: Default::default(),
            model_context: ModelContext::spawn(client),
            messages_view_model: Default::default(),
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

        frame.message_view(messages_area, &self.messages_view_model);
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
                    self.messages_view_model.handle_response(response);
                }
            }
        }
    }

    async fn submit_message(&mut self, prompt: Arc<str>) {
        self.messages_view_model
            .push_message(Message::User(prompt.clone()));

        let chat = Prompt::Chat(ChatRequest {
            prompt,
            model: Default::default(),
            history: self.messages_view_model.history(),
        });
        self.model_context
            .prompt_sender
            .send(chat)
            .await
            .expect("unable to send message")
    }
}

pub enum AppEvent {
    Submit(Arc<str>),
    Quit,
}

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

    fn handle_response(&mut self, response: Response) {
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

    fn push_message(&mut self, message: Message) {
        self.messages.push_front(message);
    }

    fn history(&self) -> Vec<Message> {
        self.messages.clone().into()
    }

    fn is_stream_empty(&self) -> bool {
        self.model_stream.is_empty()
    }

    fn clear_stream(&mut self) {
        self.model_stream.clear();
    }
}

#[ext]
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
