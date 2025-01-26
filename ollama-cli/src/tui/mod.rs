use std::{io::stdout, time::Duration};

use chat::ChatViewModel;
use crossterm::ExecutableCommand as _;
use event::{Action, EventProcessor, InputMode};
use futures::StreamExt as _;
use generate::{GenerateView, GenerateViewModel};
use model_context::ModelContext;
use modelfile::{
    modelfile::{Instruction, InstructionName},
    Modelfile,
};
use models::{ModelsView, ModelsViewModel};
use nav::{NavView, NavViewModel};
use ollama_rs::models::ModelInfo;
use popup::{PopupView, PopupViewModel};
use ratatui::{
    crossterm::event::Event,
    style::{Color, Style},
    DefaultTerminal, Frame,
};
use strum::{EnumMessage as _, VariantNames};

use crate::{
    config::Config,
    error::{Error, Result},
    lm::{Prompt, Response},
    model_definition::ModelDefinition,
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
mod popup;
mod text;
mod widgets_ext;

pub struct AppContext {
    model_context: ModelContext,
    event_processor: EventProcessor,
    popup: Option<PopupViewModel>,
    view: View,
    config: Config,
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
    pub fn handle_response_event(&mut self, event: ResponseEvent) {
        let result: Result<()> = match self {
            View::Models(models_view_model) => models_view_model.handle_response_event(event),
            View::Chat(chat_view_model) => chat_view_model.handle_response_event(event),
            View::Generate(generate_view_model) => match event {
                ResponseEvent::OllamaResponse(response) => {
                    generate_view_model.handle_response(response)
                }
                _ => Err(Error::UnexpectedResponse(event)),
            },
            View::Nav(_) => Ok(()),
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
    pub fn new(client: ollama::Client, config: Config) -> Self {
        Self {
            model_context: ModelContext::spawn(client),
            event_processor: EventProcessor::new(config.keymap.clone()),
            popup: None,
            view: Default::default(),
            config,
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
        if let Some(ref mut popup) = self.popup {
            frame.popup(frame.area(), Style::active(), popup);
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
                    self.handle_response(response).await?
                }
            }
        }
    }

    async fn handle_response(&mut self, response: Response) -> anyhow::Result<()> {
        tracing::info!(response = ?response.get_message(), "handling response");

        if let Response::LocalModels(models) = response {
            let local_modelfiles = self
                .config
                .model_cache
                .load()?
                .into_iter()
                .map(ModelDefinition::LocalCache)
                .chain(models.into_iter().map(ModelDefinition::OllamaRemote))
                .collect();

            self.view
                .handle_response_event(ResponseEvent::UpdatedModels(local_modelfiles));
            return Ok(());
        }

        let response = ResponseEvent::OllamaResponse(response);

        self.view.handle_response_event(response);
        Ok(())
    }

    /// Returns true if the event was handled
    /// and false if the app should quit.
    async fn handle_event(
        &mut self,
        terminal: &mut DefaultTerminal,
        event: AppEvent,
    ) -> anyhow::Result<bool> {
        tracing::info!("handling event",);
        match event {
            AppEvent::Submit(message) => {
                self.submit_message(message).await;
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
                if self.popup.is_some() {
                    self.popup = None;
                } else {
                    self.view = View::Nav(Default::default());
                }
                Ok(true)
            }
            AppEvent::InputMode(input_mode) => {
                self.event_processor.input_mode(input_mode);
                Ok(true)
            }
            AppEvent::EditModelInstruction {
                modelfile,
                instruction,
            } => {
                self.edit_instruction(terminal, instruction, modelfile)?;
                Ok(true)
            }
            AppEvent::EditModelfile(modelfile) => {
                self.edit_full_modelfile(terminal, modelfile)?;
                Ok(true)
            }
        }
    }

    async fn handle_input(&mut self, event: Event) -> anyhow::Result<Option<AppEvent>> {
        let action = self.event_processor.process(event);

        if let Some(ref mut popup) = self.popup {
            return Ok(popup.handle_action(action)?);
        }

        if action == Action::Popup {
            self.popup = Some(PopupViewModel::log_popup(&self.config.log_file)?);
            Ok(None)
        } else if action == Action::Help {
            self.popup = Some(PopupViewModel::keymap_popup(&self.event_processor));
            Ok(None)
        } else {
            let app_event = match &mut self.view {
                View::Chat(ref mut chat_view_model) => {
                    chat_view_model.handle_action(action).await?
                }
                View::Models(models_view_model) => models_view_model.handle_event(action).await?,
                View::Nav(nav_view_model) => nav_view_model.handle_action(action)?,
                View::Generate(generate_view_model) => generate_view_model.handle_action(action)?,
            };
            Ok(app_event)
        }
    }

    // TODO: use this function with [`modelfile`]
    fn edit_full_modelfile(
        &mut self,
        terminal: &mut DefaultTerminal,
        model_info: ModelInfo,
    ) -> anyhow::Result<()> {
        let modelfile_text = model_info.modelfile;

        let edited_modelfile = self.edit_text(terminal, modelfile_text.as_str())?;

        tracing::info!(edited_modelfile, "modelfile edited");
        Ok(())
    }

    fn edit_instruction(
        &mut self,
        terminal: &mut DefaultTerminal,
        instruction: Instruction,
        modelfile: Modelfile,
    ) -> anyhow::Result<()> {
        let instruction_name: InstructionName = (&instruction).into();

        let builder = modelfile.build_on();
        let builder = match instruction {
            Instruction::From(base_model) => {
                let old_text: String = base_model.to_string();
                let updated_text: String = self.edit_text(terminal, &old_text)?;
                builder.from(&updated_text)?
            }
            Instruction::Skip => todo!(),
            Instruction::Parameter(parameter) => todo!(),
            Instruction::Template(template) => todo!(),
            Instruction::System(system_message) => todo!(),
            Instruction::Adapter(adapter) => todo!(),
            Instruction::License(license) => todo!(),
            Instruction::Message(message) => todo!(),
        };

        let modelfile: Modelfile = builder.build()?;
        self.config.model_cache.save("test", modelfile)?;

        Ok(())
    }

    fn edit_text(&mut self, terminal: &mut DefaultTerminal, text: &str) -> anyhow::Result<String> {
        let mut edit_options = edit::Builder::default();
        let edit_options = edit_options.suffix(".tmpl");

        stdout().execute(crossterm::terminal::LeaveAlternateScreen)?;
        crossterm::terminal::disable_raw_mode()?;

        let edited_text = edit::edit_with_builder(text, edit_options)?;

        stdout().execute(crossterm::terminal::EnterAlternateScreen)?;
        crossterm::terminal::enable_raw_mode()?;
        terminal.clear()?;

        let new_modelfile: Modelfile = edited_text.parse()?;
        let json: String = serde_json::to_string(&new_modelfile)?;

        tracing::info!(%json, "modelfile edited");
        Ok(edited_text)
    }

    async fn submit_message(&mut self, prompt: Prompt) {
        self.model_context
            .prompt_sender
            .send(prompt)
            .await
            .expect("unable to send message")
    }
}

/// An event triggered from within the app.
/// Usually originally triggered from key events.
pub enum AppEvent {
    Activate(View),
    Deactivate,
    Submit(Prompt),
    EditModelInstruction {
        modelfile: Modelfile,
        instruction: Instruction,
    },
    EditModelfile(ModelInfo),
    InputMode(InputMode),
    Quit,
}

/// A response from an asyncrhonous or otherwise external source
#[derive(Debug, Clone)]
pub enum ResponseEvent {
    OllamaResponse(Response),
    UpdatedModels(Vec<ModelDefinition>),
}
