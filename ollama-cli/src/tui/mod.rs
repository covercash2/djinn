use std::{io::stdout, time::Duration};

use chat::ChatViewModel;
use crossterm::ExecutableCommand as _;
use futures::StreamExt as _;
use model_context::ModelContext;
use models::{ModelsView, ModelsViewModel};
use ollama_rs::models::ModelInfo;
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
pub mod event;
pub mod input;
pub mod messages;
mod model_context;
pub mod models;
mod widgets_ext;

pub struct AppContext {
    model_context: ModelContext,
    view: View,
}

#[derive(Clone)]
enum View {
    Chat(ChatViewModel),
    Models(ModelsViewModel),
}

impl Default for View {
    fn default() -> Self {
        View::Models(ModelsViewModel::default())
    }
}

impl View {
    pub fn handle_response(&mut self, response: Response) {
        let result = match self {
            View::Chat(ref mut chat_view_model) => chat_view_model.handle_response(response),
            View::Models(ref mut models_view_model) => models_view_model.handle_response(response),
        };

        if let Err(error) = result {
            tracing::error!(
                %error,
                "error handling response"
            );
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
            View::Models(models_view_model) => {
                frame.models_view(frame.area(), Style::default(), models_view_model)
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
                            AppEvent::EditSystemPrompt(model_info) => self.edit_model_file(&mut terminal, model_info)?,
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
            View::Models(models_view_model) => models_view_model.handle_event(event.into()).await?,
        };
        Ok(app_event)
    }

    // TODO: use this function with [`modelfile`]
    fn edit_model_file(
        &mut self,
        terminal: &mut DefaultTerminal,
        model_info: ModelInfo,
    ) -> anyhow::Result<()> {
        stdout().execute(crossterm::terminal::LeaveAlternateScreen)?;
        crossterm::terminal::disable_raw_mode()?;

        let mut edit_options = edit::Builder::default();
        let edit_options = edit_options.suffix(".tmpl");

        let _edited_modelfile = edit::edit_with_builder(model_info.modelfile, edit_options)?;

        stdout().execute(crossterm::terminal::EnterAlternateScreen)?;
        crossterm::terminal::enable_raw_mode()?;
        terminal.clear()?;
        Ok(())
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
    EditSystemPrompt(ModelInfo),
    Quit,
}
