use std::time::Duration;

use chat::ChatViewModel;
use futures::StreamExt as _;
use model_context::ModelContext;
use ratatui::{
    crossterm::event::Event,
    style::{Color, Style},
    DefaultTerminal, Frame,
};

use crate::{
    lm::{Prompt, Response},
    ollama,
    tui::chat::ChatView as _,
};

pub mod chat;
pub mod input;
pub mod messages;
mod model_context;
mod widgets_ext;

pub struct AppContext {
    model_context: ModelContext,
    view: View,
}

#[derive(Clone)]
enum View {
    Chat(ChatViewModel),
}

impl Default for View {
    fn default() -> Self {
        View::Chat(ChatViewModel::default())
    }
}

impl View {
    pub fn handle_response(&mut self, response: Response) {
        match self {
            View::Chat(ref mut chat_view_model) => chat_view_model.handle_response(response),
        }
    }
}

#[extend::ext]
impl Style {
    fn focused() -> Self {
        Style::default().fg(Color::Green)
    }

    fn active() -> Self {
        Style::default().fg(Color::Cyan)
    }
}

impl AppContext {
    pub fn new(client: ollama::Client) -> Self {
        Self {
            model_context: ModelContext::spawn(client),
            view: Default::default(),
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        match &mut self.view {
            View::Chat(ref mut chat_view_model) => {
                frame.chat_view(frame.area(), Style::default(), chat_view_model);
            }
        }
    }

    pub async fn run(mut self, mut terminal: DefaultTerminal) -> anyhow::Result<()> {
        let period = Duration::from_secs_f32(1.0 / 15.0);
        let mut interval = tokio::time::interval(period);
        let mut events = ratatui::crossterm::event::EventStream::new();
        loop {
            tokio::select! {
                _ = interval.tick() => { terminal.draw(|frame| self.draw(frame))?; },
                Some(Ok(event)) = events.next() => {
                    if let Some(app_event) = self.handle_input(event).await? {
                        match app_event {
                            AppEvent::Submit(message) => self.submit_message(message).await,
                            AppEvent::Quit => return Ok(()),
                        }
                    }
                },
                Some(response) = self.model_context.response_receiver.recv() => {
                    self.view.handle_response(response);
                }
            }
        }
    }

    async fn handle_input(&mut self, event: Event) -> anyhow::Result<Option<AppEvent>> {
        let app_event = match &mut self.view {
            View::Chat(ref mut chat_view_model) => chat_view_model.handle_event(event).await?,
        };
        Ok(app_event)
    }

    async fn submit_message(&mut self, prompt: Prompt) {
        self.model_context
            .prompt_sender
            .send(prompt)
            .await
            .expect("unable to send message")
    }
}

pub enum AppEvent {
    Submit(Prompt),
    Quit,
}
