use std::collections::HashMap;

use crossterm::event::{Event, KeyEvent};
use keymap::KeyMap;
use serde::{Deserialize, Serialize};

const DEFAULTS: &str = include_str!("../../../default_keymap.toml");

#[derive(Debug, PartialEq, Deserialize)]
pub struct EventProcessor {
    input_mode: InputMode,
    definitions: EventDefinitions,
}

impl Default for EventProcessor {
    fn default() -> Self {
        let definitions = toml::from_str(DEFAULTS).expect("should be able to load default keymaps");
        Self {
            input_mode: Default::default(),
            definitions,
        }
    }
}

impl EventProcessor {
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
            .unwrap_or(Action::Nop)
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    #[default]
    Normal,
    Edit,
}

#[derive(Debug, PartialEq, Deserialize)]
pub struct EventDefinitions(HashMap<InputMode, ActionMap>);

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

#[derive(Debug, PartialEq, Deserialize)]
pub struct ActionMap(HashMap<KeyMap, Action>);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
    use insta::assert_debug_snapshot;

    use super::*;

    #[test]
    fn load_default_keymap() {
        let event_definitions: EventDefinitions = toml::from_str(DEFAULTS).unwrap();

        assert_debug_snapshot!(event_definitions, @"");
    }
}
