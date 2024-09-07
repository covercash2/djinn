use std::sync::Arc;

use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    layout::{Constraint, Layout, Position},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, List, ListItem, Paragraph},
    DefaultTerminal, Frame,
};

use crate::ollama;

pub struct AppContext {
    client: ollama::Client,
    input_context: InputContext,
    messages: Vec<Arc<str>>,
}

#[derive(Default)]
pub struct InputContext {
    input: String,
    cursor_position: usize,
    mode: InputMode,
}

#[derive(Default)]
enum InputMode {
    #[default]
    Normal,
    Edit,
}

impl AppContext {
    pub fn new(client: ollama::Client) -> Self {
        Self {
            client,
            messages: Vec::new(),
            input_context: Default::default(),
        }
    }

    fn draw(&self, frame: &mut Frame) {
        let vertical = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(3),
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

        let input = Paragraph::new(self.input_context.input.as_str())
            .style(match self.input_context.mode {
                InputMode::Normal => Style::default(),
                InputMode::Edit => Style::default().fg(Color::Yellow),
            })
            .block(Block::bordered().title("Input"));
        frame.render_widget(input, input_area);

        match self.input_context.mode {
            InputMode::Normal => {}
            InputMode::Edit => frame.set_cursor_position(Position::new(
                input_area.x + self.input_context.cursor_position as u16 + 1,
                input_area.y + 1,
            )),
        }

        let messages: Vec<ListItem> = self
            .messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let content = Line::from(Span::raw(format!("{i}: {m}")));
                ListItem::new(content)
            })
            .collect();
        let messages = List::new(messages).block(Block::bordered().title("Messages"));
        frame.render_widget(messages, messages_area);
    }

    pub fn run(mut self, mut terminal: DefaultTerminal) -> anyhow::Result<()> {
        loop {
            terminal.draw(|frame| self.draw(frame))?;

            if let Event::Key(key) = event::read()? {
                match self.input_context.mode {
                    InputMode::Normal => match key.code {
                        KeyCode::Char('i') => {
                            self.input_context.mode = InputMode::Edit;
                        }
                        KeyCode::Char('q') => return Ok(()),
                        _ => {}
                    },
                    InputMode::Edit if key.kind == KeyEventKind::Press => match key.code {
                        KeyCode::Enter => self.submit_message(),
                        KeyCode::Char(to_insert) => self.enter_char(to_insert),
                        KeyCode::Backspace => self.delete_char(),
                        KeyCode::Esc => self.input_context.mode = InputMode::Normal,
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }

    fn submit_message(&mut self) {
        self.messages.push(self.input_context.input.clone().into());
        self.input_context.input.clear();
        self.reset_cursor();
    }

    fn reset_cursor(&mut self) {
        self.input_context.cursor_position = 0;
    }

    fn clamp_cursor(&self, new_cursor_pos: usize) -> usize {
        new_cursor_pos.clamp(0, self.input_context.input.chars().count())
    }

    fn move_cursor_left(&mut self) {
        let cursor_moved_left = self.input_context.cursor_position.saturating_sub(1);
        self.input_context.cursor_position = self.clamp_cursor(cursor_moved_left);
    }

    fn move_cursor_right(&mut self) {
        let cursor_moved_right = self.input_context.cursor_position.saturating_add(1);
        self.input_context.cursor_position = self.clamp_cursor(cursor_moved_right);
    }

    fn byte_index(&self) -> usize {
        self.input_context
            .input
            .char_indices()
            .map(|(i, _)| i)
            .nth(self.input_context.cursor_position)
            .unwrap_or(self.input_context.input.len())
    }

    fn enter_char(&mut self, new_char: char) {
        let index = self.byte_index();
        self.input_context.input.insert(index, new_char);
        self.move_cursor_right();
    }

    fn delete_char(&mut self) {
        let is_not_cursor_leftmost = self.input_context.cursor_position != 0;

        if is_not_cursor_leftmost {
            let current_index = self.input_context.cursor_position;
            let from_left_to_current_index = current_index - 1;

            let before_char_to_delete = self
                .input_context
                .input
                .chars()
                .take(from_left_to_current_index);
            let after_char_to_delete = self.input_context.input.chars().skip(current_index);

            self.input_context.input = before_char_to_delete.chain(after_char_to_delete).collect();
            self.move_cursor_left();
        }
    }
}
