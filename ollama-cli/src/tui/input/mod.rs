use std::{borrow::Cow, sync::Arc};

use ratatui::{
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
    Frame,
};

use crate::error::Result;

use super::{
    event::{Action, InputMode},
    widgets_ext::RectExt,
};

#[derive(Default, Debug, Clone)]
pub struct TextInputViewModel {
    pub input: String,
    pub cursor_position: usize,
}

#[derive(Debug, Clone)]
pub enum TextInputEvent {
    InputMode(InputMode),
    Submit(Arc<str>),
    Quit,
}

impl TextInputViewModel {
    pub fn handle_action(&mut self, action: Action) -> Result<Option<TextInputEvent>> {
        match action {
            Action::Edit => {
                return Ok(Some(TextInputEvent::InputMode(InputMode::Edit)))
            }
            Action::Quit => return Ok(Some(TextInputEvent::Quit)),
            Action::Right => self.move_cursor_right(),
            Action::Left => self.move_cursor_left(),
            Action::Beginning => self.move_cursor_to_beginning(),
            Action::End => self.move_cursor_to_end(),
            Action::RightWord => self.move_cursor_word(),
            Action::LeftWord => self.move_cursor_back(),
            Action::Enter => return Ok(Some(self.submit_message())),
            Action::Escape => {
                return Ok(Some(TextInputEvent::InputMode(InputMode::Normal)));
            }
            Action::Backspace => self.delete_char(),
            Action::Unhandled(to_insert) => self.enter_char(to_insert),
            _ => {}
        }
        Ok(None)
    }

    fn submit_message(&mut self) -> TextInputEvent {
        let message: Arc<str> = self.input.clone().into();
        self.input.clear();
        self.reset_cursor();

        TextInputEvent::Submit(message)
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

    fn move_cursor_word(&mut self) {
        self.move_cursor_to(move_cursor_word(&self.input, self.cursor_position));
    }

    fn move_cursor_back(&mut self) {
        self.move_cursor_to(move_cursor_back(&self.input, self.cursor_position))
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

fn move_cursor_word(input: &str, cursor_position: usize) -> usize {
    let (before, after) = input.split_at(cursor_position);
    after
        .find(' ')
        .map(|space_pos| before.len() + space_pos + 1)
        .unwrap_or(input.len())
}

fn move_cursor_back(input: &str, cursor_position: usize) -> usize {
    let (before, _after) = input.split_at(cursor_position);
    before
        .rfind(' ')
        .map(|space_pos| space_pos - 1)
        .unwrap_or(0)
}

#[derive(Debug, PartialEq, Clone)]
struct CursorLine<'a> {
    cursor_char: char,
    left: Cow<'a, str>,
    right: Cow<'a, str>,
}

#[extend::ext]
impl<T: AsRef<str>> T {
    /// Single out the cursor in a line of text.
    /// This function is meant to locate the cursor
    /// and the character under it
    /// for highlighting.
    fn single_out(&self, cursor_position: usize) -> CursorLine<'_> {
        if let Some((left, right)) = self.as_ref().split_at_checked(cursor_position) {
            if let Some((cursor, right)) = right.split_at_checked(1) {
                CursorLine {
                    cursor_char: cursor.chars().nth(0).unwrap_or(' '),
                    left: left.into(),
                    right: right.into(),
                }
            } else {
                CursorLine {
                    cursor_char: right.chars().nth(0).unwrap_or(' '),
                    left: left.into(),
                    right: "".into(),
                }
            }
        } else {
            CursorLine {
                cursor_char: ' ',
                left: "".into(),
                right: "".into(),
            }
        }
    }
}

#[derive(Debug, Clone)]
enum EditLine<'a> {
    Normal(Cow<'a, str>),
    WithCursor {
        string: Cow<'a, str>,
        cursor_position: usize,
    },
}

impl<'a> From<Cow<'a, str>> for EditLine<'a> {
    fn from(value: Cow<'a, str>) -> Self {
        EditLine::Normal(value)
    }
}

#[derive(Debug)]
struct EditLinesBuilder<'a> {
    consumed_chars: usize,
    lines: Vec<EditLine<'a>>,
}

impl<'a> EditLinesBuilder<'a> {
    fn new<TLine: AsRef<str>>(lines: &[TLine]) -> Self {
        EditLinesBuilder {
            consumed_chars: 0,
            lines: Vec::with_capacity(lines.len()),
        }
    }

    fn build(self) -> Vec<EditLine<'a>> {
        self.lines
    }
}

fn parse_edit_lines(input: &str, cursor_position: usize, parent_view: Rect) -> Vec<EditLine<'_>> {
    let lines = parent_view.wrap_inside(input);
    let builder = EditLinesBuilder::new(&lines);
    lines
        .into_iter()
        .fold(builder, |acc, line| {
            let EditLinesBuilder {
                consumed_chars,
                lines: mut lines_builder,
            } = acc;

            let previous_consumed = consumed_chars;
            let consumed_chars = consumed_chars + line.len() + 1;

            let line = if cursor_position >= previous_consumed && cursor_position < consumed_chars {
                let cursor_position = cursor_position - previous_consumed;
                tracing::debug!(
                    previous_consumed,
                    consumed_chars,
                    line.len = line.len(),
                    cursor_position,
                    input.len = input.len(),
                );
                EditLine::WithCursor {
                    string: line,
                    cursor_position,
                }
            } else {
                EditLine::Normal(line)
            };

            lines_builder.push(line);

            EditLinesBuilder {
                consumed_chars,
                lines: lines_builder,
            }
        })
        .build()
}

#[extend::ext]
impl Vec<EditLine<'_>> {
    fn render(&self) -> Vec<Line> {
        let style = Style::default().bg(Color::White).fg(Color::Black);

        self.iter()
            .map(|line| match line {
                EditLine::Normal(line) => Line::from(line.to_string()),
                EditLine::WithCursor {
                    string,
                    cursor_position,
                } => {
                    let cursor_line = string.single_out(*cursor_position);
                    Line::from(vec![
                        Span::from(cursor_line.left.to_string()),
                        Span::from(cursor_line.cursor_char.to_string()).style(style),
                        Span::from(cursor_line.right.to_string()),
                    ])
                }
            })
            .collect()
    }
}

#[extend::ext(name = InputView)]
pub impl<'a> Frame<'a> {
    fn input_view(&mut self, parent: Rect, style: Style, view_model: &TextInputViewModel) {
        let width = parent.width;
        let cursor_position: u16 = view_model
            .cursor_position
            .try_into()
            .inspect_err(|err| tracing::warn!(%err, "unable to convert cursor position to u16"))
            .unwrap_or(0);

        let y = cursor_position / width;

        let lines = parse_edit_lines(
            view_model.input.as_str(),
            view_model.cursor_position,
            parent,
        );

        let input = Paragraph::new(lines.render())
            .scroll((y, 0))
            .style(style)
            .block(
                Block::bordered().title(format!("cursor position: {}", view_model.cursor_position)),
            );

        self.render_widget(input, parent);
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
            left: "let ".into(),
            right: "here be light!".into(),
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
            left: "".into(),
            right: "".into(),
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
            left: line.into(),
            right: "".into(),
        };

        let result = line.single_out(cursor_position);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_move_word() {
        let line = "hello, world! i am here.";
        let cursor_position = 0;

        let expected = 7;

        let result = move_cursor_word(line, cursor_position);

        assert_eq!(result, expected);

        let cursor_position = expected;

        let expected = 14;

        let result = move_cursor_word(line, cursor_position);

        assert_eq!(result, expected);
    }
}
