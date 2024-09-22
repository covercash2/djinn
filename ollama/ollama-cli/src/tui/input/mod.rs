use std::sync::Arc;

use anyhow::Context;
use crossterm::event::KeyModifiers;
use modalkit::errors::UIResult;
use modalkit::keybindings::BindingMachine;
use modalkit::{
    actions::{Action, Editable, Jumpable, Scrollable},
    editing::{application::EmptyInfo, context::Resolve, key::KeyManager, store::Store},
    key::TerminalKey,
    prelude::RepeatType,
};
use modalkit_ratatui::textbox::{TextBox, TextBoxState};
use ratatui::{
    crossterm::event::{KeyCode, KeyEvent, KeyEventKind},
    layout::Rect,
    style::{Color, Modifier, Style, Stylize as _},
    text::{Line, Span, Text},
    widgets::{Block, Paragraph, Wrap},
    Frame,
};

use super::{widgets_ext::RectExt, AppEvent};

// #[derive(Default)]
pub struct InputViewModel {
    pub input: String,
    pub cursor_position: usize,
    pub mode: InputMode,
    pub text_box_state: TextBoxState,
    pub store: Store<EmptyInfo>,
    pub bindings: KeyManager<TerminalKey, Action, RepeatType>,
}

#[derive(Default)]
pub enum InputMode {
    Normal,
    #[default]
    Edit,
}

impl InputViewModel {
    pub fn handle_key_event(&mut self, key: KeyEvent) -> UIResult<Option<AppEvent>, EmptyInfo> {
        self.bindings.input_key(key.into());

        while let Some((act, ctx)) = self.bindings.pop() {
            let store = &mut self.store;

            let _ = match act {
                Action::Editor(act) => self.text_box_state.editor_command(&act, &ctx, store)?,
                Action::Macro(act) => self.bindings.macro_command(&act, &ctx, store)?,
                Action::Scroll(style) => self.text_box_state.scroll(&style, &ctx, store)?,
                Action::Repeat(rt) => {
                    self.bindings.repeat(rt, Some(ctx));
                    None
                }
                Action::Jump(l, dir, count) => {
                    let _ = self
                        .text_box_state
                        .jump(l, dir, ctx.resolve(&count), &ctx)?;
                    None
                }
                Action::Suspend => return Ok(Some(AppEvent::Quit)),
                Action::NoOp => None,
                _ => continue,
            };
        }

        // match self.mode {
        //     InputMode::Normal => match key.code {
        //         KeyCode::Char('i') | KeyCode::Char('a') => {
        //             self.mode = InputMode::Edit;
        //         }
        //         KeyCode::Char('q') => return Ok(Some(AppEvent::Quit)),
        //         KeyCode::Char('l') => self.move_cursor_right(),
        //         KeyCode::Char('h') => self.move_cursor_left(),
        //         KeyCode::Char('0') => self.move_cursor_to_beginning(),
        //         KeyCode::Char('$') => self.move_cursor_to_end(),
        //         _ => {}
        //     },
        //     InputMode::Edit if key.kind == KeyEventKind::Press => match key.code {
        //         KeyCode::Enter => return Ok(Some(self.submit_message())),
        //         KeyCode::Esc => self.mode = InputMode::Normal,
        //         KeyCode::Char('\\') if key.modifiers.contains(KeyModifiers::CONTROL) => {
        //             self.mode = InputMode::Normal
        //         }
        //         KeyCode::Backspace => self.delete_char(),
        //         KeyCode::Char(to_insert) => self.enter_char(to_insert),
        //         _ => {}
        //     },
        //     _ => {}
        // }

        Ok(None)
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

    fn move_cursor_to(&mut self, new_position: usize) {
        self.cursor_position = self.clamp_cursor(new_position);
    }

    fn move_cursor_to_beginning(&mut self) {
        self.move_cursor_to(0);
    }

    fn move_cursor_to_end(&mut self) {
        self.move_cursor_to(self.input.chars().count());
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

#[derive(Debug, PartialEq)]
struct CursorLine<'a> {
    cursor_char: char,
    left: &'a str,
    right: &'a str,
}

#[extend::ext]
impl str {
    fn single_out(&self, cursor_position: usize) -> CursorLine<'_> {
        if let Some((left, right)) = self.split_at_checked(cursor_position) {
            if let Some((cursor, right)) = right.split_at_checked(1) {
                CursorLine {
                    cursor_char: cursor.chars().nth(0).unwrap_or(' '),
                    left,
                    right,
                }
            } else {
                CursorLine {
                    cursor_char: right.chars().nth(0).unwrap_or(' '),
                    left,
                    right: "",
                }
            }
        } else {
            CursorLine {
                cursor_char: ' ',
                left: "",
                right: "",
            }
        }
    }
}

#[extend::ext(name = InputView)]
pub impl<'a> Frame<'a> {
    fn input_view(&mut self, parent: Rect, view_model: &mut InputViewModel) {
        let tbox = &mut view_model.text_box_state;

        let bx: TextBox = TextBox::new().block(Block::bordered())
            .left_gutter(2);

        self.render_stateful_widget(bx, parent, tbox);
        // let width = parent.width;
        // let cursor_position: u16 = view_model
        //     .cursor_position
        //     .try_into()
        //     .inspect_err(|err| tracing::warn!(%err, "unable to convert cursor position to u16"))
        //     .unwrap_or(0);
        // let x = cursor_position % width;
        // let y = cursor_position / width;
        //
        // let style = Style::default().bg(Color::White).fg(Color::Black);
        //
        // let text_lines = parent
        //     .wrap_inside(&view_model.input)
        //     .into_iter()
        //     .enumerate()
        //     .map(|(i, line)| {
        //         if i == (y as usize) {
        //             let cursor_line = line.single_out((x as usize).clamp(0, line.len()));
        //             Line::from(vec![
        //                 Span::from(cursor_line.left.to_string()),
        //                 Span::from(cursor_line.cursor_char.to_string()).style(style),
        //                 Span::from(cursor_line.right.to_string()),
        //             ])
        //         } else {
        //             Line::from(line.to_string())
        //         }
        //     })
        //     .collect::<Vec<_>>();
        //
        // let input = Paragraph::new(text_lines)
        //     .scroll((0, y))
        //     .wrap(Wrap { trim: false })
        //     .style(match view_model.mode {
        //         InputMode::Normal => Style::default(),
        //         InputMode::Edit => Style::default().fg(Color::Yellow),
        //     })
        //     .block(Block::bordered().title("Input"));
        //
        // self.render_widget(input, parent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_cursor() {
        let line = "let there be light!";
        let cursor_position: usize = 4; // t

        let expected = CursorLine {
            cursor_char: 't',
            left: "let ",
            right: "here be light!",
        };

        let result = line.single_out(cursor_position);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_split_cursor_empty_line() {
        let line = "";
        let cursor_position: usize = 0;

        let expected = CursorLine {
            cursor_char: ' ',
            left: "",
            right: "",
        };

        let result = line.single_out(cursor_position);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_split_cursor_end_of_line() {
        let line = "here:";
        let cursor_position: usize = 5;

        let expected = CursorLine {
            cursor_char: ' ',
            left: line,
            right: "",
        };

        let result = line.single_out(cursor_position);

        assert_eq!(result, expected);
    }
}
