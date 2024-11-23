use std::{io::stdout, time::Duration};

use chat::ChatViewModel;
use crossterm::ExecutableCommand as _;
use event::{Action, EventProcessor, InputMode};
use futures::StreamExt as _;
use generate::{GenerateView, GenerateViewModel};
use model_context::ModelContext;
use models::{ModelsView, ModelsViewModel};
use nav::{NavView, NavViewModel};
use ollama_rs::models::ModelInfo;
use ratatui::{
    crossterm::event::Event,
    style::{Color, Style},
    DefaultTerminal, Frame,
};
use strum::VariantNames;

use crate::{
    error::Result,
    lm::{Prompt, Response},
    ollama,
    tui::chat::ChatView as _,
};

pub mod chat;
pub mod event;
pub mod generate;
pub mod input;
pub mod messages;
mod model_context;
pub mod models;
mod nav;
mod widgets_ext;

pub struct AppContext {
    model_context: ModelContext,
    event_processor: EventProcessor,
    view: View,
}

#[derive(Clone, Debug, strum::EnumString, strum::EnumDiscriminants)]
#[strum_discriminants(derive(VariantNames))]
#[strum_discriminants(name(ViewName))]
#[strum_discriminants(strum(serialize_all = "lowercase"))]
#[strum(serialize_all = "lowercase")]
pub enum View {
    Models(ModelsViewModel),
    Chat(ChatViewModel),
    Generate(GenerateViewModel),
    Nav(NavViewModel),
}

impl Default for View {
    fn default() -> Self {
        View::Nav(Default::default())
    }
}

impl View {
    pub fn handle_response(&mut self, response: Response) {
        let result = match self {
            View::Chat(ref mut chat_view_model) => chat_view_model.handle_response(response),
            View::Models(ref mut models_view_model) => models_view_model.handle_response(response),
            View::Generate(ref mut view_model) => view_model.handle_response(response),
            View::Nav(_nav_view_model) => Ok(()),
        };

        if let Err(error) = result {
            tracing::error!(
                %error,
                "error handling response"
            );
        }
    }

    pub async fn init(&mut self) -> Result<Option<AppEvent>> {
        match self {
            View::Chat(_chat_view_model) => Ok(None),
            View::Models(models_view_model) => {
                models_view_model.handle_event(Action::Refresh).await
            }
            View::Nav(_nav_view_model) => Ok(None),
            View::Generate(_generate_view_model) => Ok(None),
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
            event_processor: Default::default(),
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
            View::Nav(nav_view_model) => {
                frame.nav_view(frame.area(), Style::active(), nav_view_model)
            }
            View::Generate(generate_view_model) => {
                frame.generate_view(frame.area(), Style::default(), generate_view_model)
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
                        let cont = self.handle_event(&mut terminal, app_event).await?;
                        if !cont {
                            return Ok(());
                        }
                    }
                },
                Some(response) = self.model_context.response_receiver.recv() => {
                    self.view.handle_response(response);
                }
            }
        }
    }

    /// Returns true if the event was handled
    /// and false if the app should quit.
    async fn handle_event(
        &mut self,
        terminal: &mut DefaultTerminal,
        event: AppEvent,
    ) -> anyhow::Result<bool> {
        match event {
            AppEvent::Submit(message) => {
                self.submit_message(message).await;
                Ok(true)
            }
            AppEvent::EditSystemPrompt(model_info) => {
                self.edit_model_file(terminal, model_info)?;
                Ok(true)
            }
            AppEvent::Quit => Ok(false),
            AppEvent::Activate(view) => {
                self.view = view;
                if let Some(event) = self.view.init().await? {
                    // necessary because of async recursion
                    Box::pin(self.handle_event(terminal, event)).await
                } else {
                    Ok(true)
                }
            }
            AppEvent::Deactivate => {
                self.view = View::Nav(Default::default());
                Ok(true)
            }
            AppEvent::InputMode(input_mode) => {
                self.event_processor.input_mode(input_mode);
                Ok(true)
            }
        }
    }

    async fn handle_input(&mut self, event: Event) -> anyhow::Result<Option<AppEvent>> {
        let action = self.event_processor.process(event);
        let app_event = match &mut self.view {
            View::Chat(ref mut chat_view_model) => chat_view_model.handle_action(action).await?,
            View::Models(models_view_model) => models_view_model.handle_event(action).await?,
            View::Nav(nav_view_model) => nav_view_model.handle_action(action)?,
            View::Generate(generate_view_model) => generate_view_model.handle_action(action)?,
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
    Activate(View),
    Deactivate,
    Submit(Prompt),
    EditSystemPrompt(ModelInfo),
    InputMode(InputMode),
    Quit,
}
