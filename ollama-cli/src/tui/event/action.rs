use std::collections::HashMap;

use keymap::KeyMap;
use serde::{Deserialize, Serialize};
use crate::{error::Result, tui::AppEvent};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionDefinition {
    action: Action,
    description: String,
}

impl From<Action> for ActionDefinition {
    fn from(action: Action) -> Self {
        ActionDefinition {
            action,
            description: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ActionMap(pub HashMap<KeyMap, Action>);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, strum::Display)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Beginning,
    End,
    Left,
    Right,
    Up,
    Down,
    Edit,
    LeftWord,
    RightWord,
    Refresh,
    Popup,
    Help,
    Enter,
    Escape,
    Backspace,
    Quit,
    #[serde(skip)]
    Unhandled(char),
    Nop,
}

pub trait ActionHandler {
    type Event;
    fn handle_action(&mut self, action: Action) -> Result<Option<AppEvent>>;
}
