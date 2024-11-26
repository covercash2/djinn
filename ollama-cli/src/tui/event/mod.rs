use std::collections::HashMap;

use crossterm::event::{Event, KeyCode, KeyEvent};
use keymap::KeyMap;
use serde::{Deserialize, Serialize};
use strum::EnumIter;

const DEFAULTS: &str = include_str!("../../../default_keymap.toml");

#[derive(Debug, PartialEq, Deserialize)]
pub struct EventProcessor {
    pub input_mode: InputMode,
    pub definitions: EventDefinitions,
}

impl EventProcessor {
    pub fn new(definitions: EventDefinitions) -> Self {
        EventProcessor {
            input_mode: Default::default(),
            definitions,
        }
    }

    pub fn input_mode(&mut self, input_mode: InputMode) {
        self.input_mode = input_mode;
    }

    pub fn process(&self, event: Event) -> Action {
        match event {
            Event::FocusGained
            | Event::FocusLost
            | Event::Mouse(_)
            | Event::Paste(_)
            | Event::Resize(_, _) => Action::Nop,
            Event::Key(key_event) => self.process_key_event(key_event),
        }
    }

    pub fn process_key_event(&self, event: KeyEvent) -> Action {
        self.definitions
            .0
            .get(&self.input_mode)
            .and_then(|map| map.0.get(&KeyMap::from(event)))
            .copied()
            .unwrap_or(match event.code {
                KeyCode::Char(c) => Action::Unhandled(c),
                _ => Action::Nop,
            })
    }
}

#[derive(
    Default,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    EnumIter,
    strum::Display,
)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    #[default]
    Normal,
    Edit,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EventDefinitions(pub HashMap<InputMode, ActionMap>);

impl Default for EventDefinitions {
    fn default() -> Self {
        toml::from_str(DEFAULTS).expect("should be able to load default keymaps")
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_default_keymap() {
        let _event_definitions: EventDefinitions = toml::from_str(DEFAULTS).unwrap();
    }
}
