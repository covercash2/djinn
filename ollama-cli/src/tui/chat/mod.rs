use std::sync::Arc;

use crossterm::event::{Event, KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style, Stylize as _},
    text::{Line, Text},
    widgets::Paragraph,
    Frame,
};

use crate::{
    error::Result,
    lm::{Prompt, Response},
    ollama::chat::{ChatRequest, Message},
};

use super::{
    input::{InputMode, InputView as _, TextInputEvent, TextInputViewModel},
    messages::{view::MessagesView as _, MessagesEvent, MessagesViewModel},
    AppEvent, StyleExt as _,
};

#[derive(Default, Clone, Debug)]
pub struct ChatViewModel {
    text_input: TextInputViewModel,
    messages: MessagesViewModel,
    active_view: Option<Pane>,
    focused_view: Pane,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum Pane {
    #[default]
    Input,
    Messages,
}

impl Pane {
    fn next(self) -> Pane {
        match self {
            Pane::Input => Pane::Messages,
            Pane::Messages => Pane::Input,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ChatEvent {
    Activate(Pane),
    Deactivate,
    NextView,
    Submit(Arc<str>),
    Quit,
}

impl From<MessagesEvent> for ChatEvent {
    fn from(value: MessagesEvent) -> Self {
        match value {
            MessagesEvent::Quit => ChatEvent::Quit,
        }
    }
}

impl From<TextInputEvent> for ChatEvent {
    fn from(value: TextInputEvent) -> Self {
        match value {
            TextInputEvent::Submit(message) => ChatEvent::Submit(message),
            TextInputEvent::Quit => ChatEvent::Deactivate,
        }
    }
}

impl ChatViewModel {
    pub fn handle_response(&mut self, response: Response) -> Result<()> {
        self.messages.handle_response(response)
    }

    pub async fn handle_event(&mut self, event: Event) -> anyhow::Result<Option<AppEvent>> {
        let chat_event: Option<ChatEvent> = match event {
            Event::FocusGained
            | Event::FocusLost
            | Event::Mouse(_)
            | Event::Paste(_)
            | Event::Resize(_, _) => None,
            Event::Key(key_event) => self.handle_key_event(key_event).await?,
        };

        Ok(chat_event.and_then(|event| self.handle_chat_event(event)))
    }

    fn handle_chat_event(&mut self, event: ChatEvent) -> Option<AppEvent> {
        match event {
            ChatEvent::Activate(pane) => {
                self.active_view = Some(pane);
                None
            }
            ChatEvent::Deactivate => {
                self.active_view = None;
                None
            }
            ChatEvent::NextView => {
                self.focused_view = self.focused_view.next();
                None
            }
            ChatEvent::Submit(prompt) => {
                self.messages.push_message(Message::User(prompt.clone()));
                let prompt = Prompt::Chat(ChatRequest {
                    prompt,
                    model: Default::default(),
                    history: self.messages.history(),
                });
                Some(AppEvent::Submit(prompt))
            }
            ChatEvent::Quit => Some(AppEvent::Quit),
        }
    }

    pub async fn handle_key_event(&mut self, event: KeyEvent) -> anyhow::Result<Option<ChatEvent>> {
        if let Some(active_view) = self.active_view {
            match active_view {
                Pane::Input => {
                    let app_event: Option<ChatEvent> =
                        self.text_input.handle_key_event(event).map(Into::into);
                    Ok(app_event)
                }
                Pane::Messages => {
                    let app_event: Option<ChatEvent> =
                        self.messages.handle_key_event(event).map(Into::into);

                    Ok(app_event)
                }
            }
        } else {
            match event.code {
                KeyCode::Char('q') => Ok(Some(ChatEvent::Quit)),
                KeyCode::Char('j')
                | KeyCode::Char('k')
                | KeyCode::Up
                | KeyCode::Down
                | KeyCode::Char('h')
                | KeyCode::Char('l') => Ok(Some(ChatEvent::NextView)),
                KeyCode::Enter => Ok(Some(ChatEvent::Activate(self.focused_view))),
                _ => Ok(None),
            }
        }
    }
}

#[extend::ext(name = ChatView)]
pub impl<'a> Frame<'a> {
    fn chat_view(&mut self, parent: Rect, style: Style, view_model: &mut ChatViewModel) {
        let vertical = Layout::vertical([
            Constraint::Length(1),
            Constraint::Max(5),
            Constraint::Min(1),
        ]);

        let [help_area, input_area, messages_area] = vertical.areas(parent);

        let (msg, style) = match view_model.text_input.mode {
            InputMode::Normal => (
                vec![
                    "Press ".into(),
                    "q".bold(),
                    " to exit, ".into(),
                    "e".bold(),
                    " to start editing.".bold(),
                ],
                style.add_modifier(Modifier::RAPID_BLINK),
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
        self.render_widget(help_message, help_area);

        let input_style = if view_model.focused_view == Pane::Input {
            if let Some(Pane::Input) = view_model.active_view {
                Style::active()
            } else {
                Style::focused()
            }
        } else {
            Style::default()
        };
        self.input_view(input_area, input_style, &view_model.text_input);

        let messages_style = if view_model.focused_view == Pane::Messages {
            if let Some(Pane::Messages) = view_model.active_view {
                Style::active()
            } else {
                Style::focused()
            }
        } else {
            Style::default()
        };
        self.messages_view(messages_area, messages_style, &mut view_model.messages);
    }
}
