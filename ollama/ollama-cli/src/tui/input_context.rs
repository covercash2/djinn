use std::sync::Arc;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind};

use super::AppEvent;

#[derive(Default)]
pub struct InputContext {
    pub input: String,
    pub cursor_position: usize,
    pub mode: InputMode,
}

#[derive(Default)]
pub enum InputMode {
    #[default]
    Normal,
    Edit,
}

impl InputContext {
    pub fn handle_key_event(&mut self, key: KeyEvent) -> Option<AppEvent> {
        match self.mode {
            InputMode::Normal => match key.code {
                KeyCode::Char('i') | KeyCode::Char('a') => {
                    self.mode = InputMode::Edit;
                }
                KeyCode::Char('q') => return Some(AppEvent::Quit),
                _ => {}
            },
            InputMode::Edit if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Enter => return Some(self.submit_message()),
                KeyCode::Char(to_insert) => self.enter_char(to_insert),
                KeyCode::Backspace => self.delete_char(),
                KeyCode::Esc => self.mode = InputMode::Normal,
                _ => {}
            },
            _ => {}
        }

        None
    }

    fn submit_message(&mut self) -> AppEvent {
        let message: Arc<str> = self.input.clone().into();
        self.input.clear();
        self.reset_cursor();

        AppEvent::Submit(message)
    }

    fn reset_cursor(&mut self) {
        self.cursor_position = 0;
    }

    fn clamp_cursor(&self, new_cursor_pos: usize) -> usize {
        new_cursor_pos.clamp(0, self.input.chars().count())
    }

    fn move_cursor_left(&mut self) {
        let cursor_moved_left = self.cursor_position.saturating_sub(1);
        self.cursor_position = self.clamp_cursor(cursor_moved_left);
    }

    fn move_cursor_right(&mut self) {
        let cursor_moved_right = self.cursor_position.saturating_add(1);
        self.cursor_position = self.clamp_cursor(cursor_moved_right);
    }

    fn byte_index(&self) -> usize {
        self.input
            .char_indices()
            .map(|(i, _)| i)
            .nth(self.cursor_position)
            .unwrap_or(self.input.len())
    }

    fn enter_char(&mut self, new_char: char) {
        let index = self.byte_index();
        self.input.insert(index, new_char);
        self.move_cursor_right();
    }

    fn delete_char(&mut self) {
        let is_not_cursor_leftmost = self.cursor_position != 0;

        if is_not_cursor_leftmost {
            let current_index = self.cursor_position;
            let from_left_to_current_index = current_index - 1;

            let before_char_to_delete = self.input.chars().take(from_left_to_current_index);
            let after_char_to_delete = self.input.chars().skip(current_index);

            self.input = before_char_to_delete.chain(after_char_to_delete).collect();
            self.move_cursor_left();
        }
    }
}
