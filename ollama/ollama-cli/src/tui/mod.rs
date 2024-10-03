use std::{sync::Arc, time::Duration};

use crossterm::event::{KeyCode, KeyEvent};
use futures::StreamExt as _;
use input::{InputMode, InputView, TextInputEvent, TextInputViewModel};
use messages::{MessagesEvent, MessagesViewModel};
use model_context::ModelContext;
use ratatui::{
    crossterm::event::Event,
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Text},
    widgets::Paragraph,
    DefaultTerminal, Frame,
};

use crate::{
    lm::Prompt,
    ollama::{
        self,
        chat::{ChatRequest, Message},
    },
    tui::messages::MessagesView as _,
};

mod input;
mod messages;
mod model_context;
mod widgets_ext;

pub struct AppContext {
    input_context: TextInputViewModel,
    model_context: ModelContext,
    messages_view_model: MessagesViewModel,
    active_view: Option<ChatPanes>,
    focused_view: ChatPanes,
}

#[derive(Default, Clone, Copy, PartialEq)]
pub enum ChatPanes {
    #[default]
    Input,
    Messages,
}

impl ChatPanes {
    fn next(self) -> ChatPanes {
        match self {
            ChatPanes::Input => ChatPanes::Messages,
            ChatPanes::Messages => ChatPanes::Input,
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
            input_context: Default::default(),
            model_context: ModelContext::spawn(client),
            messages_view_model: Default::default(),
            active_view: Default::default(),
            focused_view: Default::default(),
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        let vertical = Layout::vertical([
            Constraint::Length(1),
            Constraint::Max(5),
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

        let input_style = if self.focused_view == ChatPanes::Input {
            if let Some(ChatPanes::Input) = self.active_view {
                Style::active()
            } else {
                Style::focused()
            }
        } else {
            Style::default()
        };
        frame.input_view(input_area, input_style, &self.input_context);

        let messages_style = if self.focused_view == ChatPanes::Messages {
            if let Some(ChatPanes::Messages) = self.active_view {
                Style::active()
            } else {
                Style::focused()
            }
        } else {
            Style::default()
        };
        frame.messages_view(messages_area, messages_style, &mut self.messages_view_model);
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
                            AppEvent::Activate(view) => self.active_view = Some(view),
                            AppEvent::Deactivate => self.active_view = None,
                            AppEvent::NextView => self.focused_view = self.focused_view.next(),
                        }
                    }
                },
                Some(response) = self.model_context.response_receiver.recv() => {
                    self.messages_view_model.handle_response(response);
                }
            }
        }
    }

    async fn handle_input(&mut self, event: Event) -> anyhow::Result<Option<AppEvent>> {
        match event {
            Event::FocusGained
            | Event::FocusLost
            | Event::Mouse(_)
            | Event::Paste(_)
            | Event::Resize(_, _) => Ok(None),
            Event::Key(key_event) => self.handle_key_event(key_event).await,
        }
    }

    async fn handle_key_event(&mut self, event: KeyEvent) -> anyhow::Result<Option<AppEvent>> {
        if let Some(active_view) = self.active_view {
            match active_view {
                ChatPanes::Input => {
                    let app_event: Option<AppEvent> = self
                        .input_context
                        .handle_key_event(event)
                        .map(|input_event| match input_event {
                            TextInputEvent::Submit(message) => AppEvent::Submit(message),
                            TextInputEvent::Quit => AppEvent::Deactivate,
                        });
                    Ok(app_event)
                }
                ChatPanes::Messages => {
                    let app_event: Option<AppEvent> = self.messages_view_model.handle_key_event(event)
                        .map(Into::into);

                    Ok(app_event)
                },
            }
        } else {
            match event.code {
                KeyCode::Char('q') => Ok(Some(AppEvent::Quit)),
                KeyCode::Char('j')
                | KeyCode::Char('k')
                | KeyCode::Up
                | KeyCode::Down
                | KeyCode::Char('h')
                | KeyCode::Char('l') => Ok(Some(AppEvent::NextView)),
                KeyCode::Enter => Ok(Some(AppEvent::Activate(self.focused_view))),
                _ => Ok(None),
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
    Activate(ChatPanes),
    Deactivate,
    NextView,
    Submit(Arc<str>),
    Quit,
}

impl From<MessagesEvent> for AppEvent {
    fn from(value: MessagesEvent) -> Self {
        match value {
            MessagesEvent::Quit => AppEvent::Deactivate,
        }
    }
}
