use std::sync::Arc;

use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Style,
    Frame,
};

use crate::{
    error::Result,
    lm::Prompt,
    ollama::chat::{ChatRequest, Message},
};

use super::{
    event::{Action, InputMode},
    input::{InputView as _, TextInputEvent, TextInputViewModel},
    messages::{view::MessagesView as _, MessagesEvent, MessagesViewModel},
    AppEvent, ResponseEvent, StyleExt as _,
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
    InputMode(InputMode),
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
            TextInputEvent::InputMode(input_mode) => ChatEvent::InputMode(input_mode),
        }
    }
}

impl ChatViewModel {
    pub fn handle_response_event(&mut self, event: ResponseEvent) -> Result<()> {
        self.messages.handle_response_event(event)
    }

    pub async fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>> {
        if let Some(active_view) = self.active_view {
            match active_view {
                Pane::Input => {
                    let chat_event: Option<ChatEvent> =
                        self.text_input.handle_action(action)?.map(Into::into);
                    if let Some(chat_event) = chat_event {
                        Ok(self.handle_chat_event(chat_event))
                    } else {
                        Ok(None)
                    }
                }
                Pane::Messages => {
                    let chat_event: Option<ChatEvent> =
                        self.messages.handle_action(action).map(Into::into);
                    if let Some(chat_event) = chat_event {
                        Ok(self.handle_chat_event(chat_event))
                    } else {
                        Ok(None)
                    }
                }
            }
        } else {
            let chat_event = match action {
                Action::Quit => Some(ChatEvent::Quit),
                Action::Up | Action::Down | Action::Left | Action::Right => {
                    Some(ChatEvent::NextView)
                }
                Action::Enter => Some(ChatEvent::Activate(self.focused_view)),
                _ => None,
            };
            if let Some(chat_event) = chat_event {
                Ok(self.handle_chat_event(chat_event))
            } else {
                Ok(None)
            }
        }
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
            ChatEvent::Quit => Some(AppEvent::Deactivate),
            ChatEvent::InputMode(input_mode) => Some(AppEvent::InputMode(input_mode)),
        }
    }
}

#[extend::ext(name = ChatView)]
pub impl<'a> Frame<'a> {
    fn chat_view(&mut self, parent: Rect, style: Style, view_model: &mut ChatViewModel) {
        let vertical = Layout::vertical([Constraint::Max(5), Constraint::Min(1)]);

        let [input_area, messages_area] = vertical.areas(parent);

        let input_style = if view_model.focused_view == Pane::Input {
            if let Some(Pane::Input) = view_model.active_view {
                Style::active()
            } else {
                Style::focused()
            }
        } else {
            style
        };
        self.input_view(input_area, input_style, &view_model.text_input);

        let messages_style = if view_model.focused_view == Pane::Messages {
            if let Some(Pane::Messages) = view_model.active_view {
                Style::active()
            } else {
                Style::focused()
            }
        } else {
            style
        };
        self.messages_view(messages_area, messages_style, &mut view_model.messages);
    }
}
