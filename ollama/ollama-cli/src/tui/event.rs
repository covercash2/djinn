use crossterm::event::{Event, KeyCode, KeyEvent};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Action {
    Left,
    Right,
    Up,
    Down,
    LeftWord,
    RightWord,
    Refresh,
    Enter,
    Quit,
    #[default]
    Unhandled,
}

impl From<Event> for Action {
    fn from(event: Event) -> Self {
        match event {
            Event::FocusGained
            | Event::FocusLost
            | Event::Mouse(_)
            | Event::Paste(_)
            | Event::Resize(_, _) => Action::Unhandled,
            Event::Key(key_event) => key_event.into(),
        }
    }
}

impl From<KeyEvent> for Action {
    fn from(key_event: KeyEvent) -> Self {
        match key_event.code {
            KeyCode::Char('q') => Action::Quit,
            KeyCode::Char('j') => Action::Down,
            KeyCode::Char('k') => Action::Up,
            KeyCode::Up => Action::Up,
            KeyCode::Down => Action::Down,
            KeyCode::Char('h') => Action::Left,
            KeyCode::Char('l') => Action::Right,
            KeyCode::Char('r') => Action::Refresh,
            KeyCode::Enter => Action::Enter,
            _ => Action::Unhandled,
        }
    }
}
