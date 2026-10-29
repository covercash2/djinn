//! The multi-line input buffer (design D7).
//!
//! Text is held as logical lines with a cursor counted in characters. Display layout (soft wrapping
//! at the box width) is computed on demand by [`InputBuffer::layout`]. Input typed by the user is the
//! one kind of text that is not run through `sanitise`: it only ever goes to the model, never back
//! to the terminal as raw bytes, but control characters are still refused here.

use unicode_width::UnicodeWidthChar;

/// Where the cursor is drawn and what to draw, for a box of a given width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub rows: Vec<String>,
    /// Row and column (in terminal columns) of the cursor within `rows`.
    pub cursor: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct InputBuffer {
    lines: Vec<String>,
    row: usize,
    col: usize,
    history: Vec<String>,
    /// Which history entry is shown, if the user is browsing.
    browsing: Option<usize>,
    /// What was typed before browsing started.
    draft: Option<String>,
}

impl Default for InputBuffer {
    fn default() -> Self {
        Self {
            lines: vec![String::new()],
            row: 0,
            col: 0,
            history: Vec::new(),
            browsing: None,
            draft: None,
        }
    }
}

fn byte_index(line: &str, chars: usize) -> usize {
    line.char_indices()
        .nth(chars)
        .map_or(line.len(), |(index, _)| index)
}

fn char_count(line: &str) -> usize {
    line.chars().count()
}

impl InputBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// The whole text, lines joined by `\n`.
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// The cursor as (line, column), the column counted in characters.
    pub fn cursor(&self) -> (usize, usize) {
        (self.row, self.col)
    }

    /// Replaces the whole text, with the cursor at its end.
    pub fn replace(&mut self, text: &str) {
        self.leave_history();
        self.set_text(text);
    }

    pub fn is_empty(&self) -> bool {
        self.lines.len() == 1 && self.lines[0].is_empty()
    }

    /// Nothing but whitespace (so there is nothing to send).
    pub fn is_blank(&self) -> bool {
        self.lines.iter().all(|line| line.trim().is_empty())
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    fn line(&self) -> &str {
        &self.lines[self.row]
    }

    fn leave_history(&mut self) {
        self.browsing = None;
        self.draft = None;
    }

    pub fn insert_char(&mut self, c: char) {
        if c == '\n' {
            self.newline();
            return;
        }
        if c.is_control() {
            return;
        }
        self.leave_history();
        let at = byte_index(self.line(), self.col);
        self.lines[self.row].insert(at, c);
        self.col += 1;
    }

    /// Inserts pasted text as if typed, but never sends it. `\r\n` and `\r` count as one newline;
    /// tabs become four spaces; other controls are dropped.
    pub fn insert_str(&mut self, text: &str) {
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '\r' => {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    self.newline();
                }
                '\t' => "    ".chars().for_each(|c| self.insert_char(c)),
                c => self.insert_char(c),
            }
        }
    }

    pub fn newline(&mut self) {
        self.leave_history();
        let at = byte_index(self.line(), self.col);
        let rest = self.lines[self.row].split_off(at);
        self.lines.insert(self.row + 1, rest);
        self.row += 1;
        self.col = 0;
    }

    pub fn backspace(&mut self) {
        self.leave_history();
        if self.col > 0 {
            let at = byte_index(self.line(), self.col - 1);
            self.lines[self.row].remove(at);
            self.col -= 1;
        } else if self.row > 0 {
            let line = self.lines.remove(self.row);
            self.row -= 1;
            self.col = char_count(self.line());
            self.lines[self.row].push_str(&line);
        }
    }

    pub fn delete(&mut self) {
        self.leave_history();
        if self.col < char_count(self.line()) {
            let at = byte_index(self.line(), self.col);
            self.lines[self.row].remove(at);
        } else if self.row + 1 < self.lines.len() {
            let next = self.lines.remove(self.row + 1);
            self.lines[self.row].push_str(&next);
        }
    }

    pub fn left(&mut self) {
        if self.col > 0 {
            self.col -= 1;
        } else if self.row > 0 {
            self.row -= 1;
            self.col = char_count(self.line());
        }
    }

    pub fn right(&mut self) {
        if self.col < char_count(self.line()) {
            self.col += 1;
        } else if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.col = 0;
        }
    }

    pub fn home(&mut self) {
        self.col = 0;
    }

    pub fn end(&mut self) {
        self.col = char_count(self.line());
    }

    /// Moves up a line; on the first line recalls an earlier message instead.
    pub fn up(&mut self) {
        if self.row > 0 {
            self.row -= 1;
            self.col = self.col.min(char_count(self.line()));
            return;
        }
        let next = match self.browsing {
            None if self.history.is_empty() => return,
            None => {
                self.draft = Some(self.text());
                self.history.len() - 1
            }
            Some(0) => return,
            Some(index) => index - 1,
        };
        self.show_history(next);
    }

    /// Moves down a line; on the last line steps forward through the history.
    pub fn down(&mut self) {
        if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.col = self.col.min(char_count(self.line()));
            return;
        }
        let Some(index) = self.browsing else { return };
        if index + 1 < self.history.len() {
            self.show_history(index + 1);
        } else {
            let draft = self.draft.take().unwrap_or_default();
            self.set_text(&draft);
            self.browsing = None;
        }
    }

    fn show_history(&mut self, index: usize) {
        let text = self.history[index].clone();
        let draft = self.draft.take();
        self.set_text(&text);
        self.browsing = Some(index);
        self.draft = draft;
    }

    fn set_text(&mut self, text: &str) {
        self.lines = text.split('\n').map(str::to_owned).collect();
        self.row = self.lines.len() - 1;
        self.col = char_count(self.line());
    }

    /// Deletes from the cursor to the end of the line (Ctrl-K).
    pub fn kill_to_end(&mut self) {
        self.leave_history();
        let at = byte_index(self.line(), self.col);
        self.lines[self.row].truncate(at);
    }

    /// Deletes from the start of the line to the cursor (Ctrl-U).
    pub fn kill_to_start(&mut self) {
        self.leave_history();
        let at = byte_index(self.line(), self.col);
        self.lines[self.row].replace_range(..at, "");
        self.col = 0;
    }

    /// Deletes the word before the cursor (Ctrl-W).
    pub fn delete_word_back(&mut self) {
        self.leave_history();
        let chars: Vec<char> = self.line().chars().collect();
        let mut start = self.col;
        while start > 0 && chars[start - 1].is_whitespace() {
            start -= 1;
        }
        while start > 0 && !chars[start - 1].is_whitespace() {
            start -= 1;
        }
        let from = byte_index(self.line(), start);
        let to = byte_index(self.line(), self.col);
        self.lines[self.row].replace_range(from..to, "");
        self.col = start;
    }

    /// Takes the text out for sending and remembers it for recall. Returns `None` when blank.
    pub fn take(&mut self) -> Option<String> {
        if self.is_blank() {
            return None;
        }
        let text = self.text();
        if self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }
        self.clear();
        Some(text)
    }

    pub fn clear(&mut self) {
        *self = Self {
            history: std::mem::take(&mut self.history),
            ..Self::default()
        };
    }

    /// Soft-wraps the text to `columns` and finds the cursor.
    pub fn layout(&self, columns: usize) -> Layout {
        let limit = columns.max(2);
        let mut rows = Vec::new();
        let mut cursor = (0, 0);
        for (index, line) in self.lines.iter().enumerate() {
            let first_row = rows.len();
            let mut current = String::new();
            let mut used = 0;
            let mut cursor_here = None;
            for (position, c) in line.chars().enumerate() {
                let w = c.width().unwrap_or(0);
                if used > 0 && used + w > limit {
                    rows.push(std::mem::take(&mut current));
                    used = 0;
                }
                if index == self.row && position == self.col {
                    cursor_here = Some((rows.len(), used));
                }
                current.push(c);
                used += w;
            }
            if index == self.row && self.col == char_count(line) {
                // At the end of the line; a full row pushes the cursor onto a fresh one.
                cursor_here = Some(if used >= limit {
                    (rows.len() + 1, 0)
                } else {
                    (rows.len(), used)
                });
            }
            rows.push(current);
            if let Some((row, col)) = cursor_here {
                cursor = (row, col);
                if row >= rows.len() {
                    rows.push(String::new());
                }
            }
            debug_assert!(rows.len() > first_row);
        }
        Layout { rows, cursor }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(text: &str) -> InputBuffer {
        let mut input = InputBuffer::new();
        input.insert_str(text);
        input
    }

    #[test]
    fn typing_and_editing() {
        let mut input = typed("hello");
        assert_eq!(input.text(), "hello");
        input.backspace();
        input.left();
        input.left();
        input.insert_char('X');
        assert_eq!(input.text(), "heXll");
        input.delete();
        assert_eq!(input.text(), "heXl");
        input.home();
        input.insert_char('>');
        input.end();
        input.insert_char('<');
        assert_eq!(input.text(), ">heXl<");
    }

    #[test]
    fn editing_works_on_multibyte_characters() {
        let mut input = typed("héllo 你好");
        input.backspace();
        input.left();
        input.insert_char('é');
        assert_eq!(input.text(), "héllo é你");
        input.home();
        input.delete();
        assert_eq!(input.text(), "éllo é你");
    }

    #[test]
    fn newlines_split_and_backspace_joins() {
        let mut input = typed("ab");
        input.left();
        input.newline();
        assert_eq!(input.text(), "a\nb");
        assert_eq!(input.line_count(), 2);
        input.backspace();
        assert_eq!(input.text(), "ab");
        input.end();
        input.newline();
        input.delete();
        assert_eq!(input.text(), "ab\n");
        input.left();
        input.delete();
        assert_eq!(input.text(), "ab");
    }

    #[test]
    fn a_paste_with_newlines_fills_the_box_and_sends_nothing() {
        let mut input = InputBuffer::new();
        input.insert_str("one\r\ntwo\rthree\tx\u{1b}[31m");
        assert_eq!(input.text(), "one\ntwo\nthree    x[31m");
        assert_eq!(input.line_count(), 3);
    }

    #[test]
    fn arrows_move_across_lines() {
        let mut input = typed("abc\nde");
        input.up();
        assert_eq!(input.layout(20).cursor, (0, 2));
        input.end();
        input.right();
        assert_eq!(input.layout(20).cursor, (1, 0));
        input.left();
        assert_eq!(input.layout(20).cursor, (0, 3));
        input.down();
        assert_eq!(
            input.layout(20).cursor,
            (1, 2),
            "column is clamped to the shorter line"
        );
    }

    #[test]
    fn kill_and_word_delete() {
        let mut input = typed("one two three");
        input.delete_word_back();
        assert_eq!(input.text(), "one two ");
        input.delete_word_back();
        assert_eq!(input.text(), "one ");
        input.left();
        input.left();
        input.kill_to_end();
        assert_eq!(input.text(), "on");
        input.kill_to_start();
        assert_eq!(input.text(), "");
    }

    #[test]
    fn replace_sets_the_text_and_puts_the_cursor_at_the_end() {
        let mut input = typed("old\ntext");
        input.replace("/reset");
        assert_eq!(input.text(), "/reset");
        assert_eq!(input.cursor(), (0, 6));
        input.replace("a\nbc");
        assert_eq!(input.cursor(), (1, 2));
    }

    #[test]
    fn blank_input_is_not_sent() {
        let mut input = typed("  \n \t ");
        assert!(input.is_blank());
        assert_eq!(input.take(), None);
        assert!(!input.is_empty(), "the blanks stay for the user to see");
    }

    #[test]
    fn take_returns_the_text_and_clears_the_box() {
        let mut input = typed("hello\nworld");
        assert_eq!(input.take().as_deref(), Some("hello\nworld"));
        assert!(input.is_empty());
    }

    #[test]
    fn history_is_recalled_only_from_the_first_and_last_lines() {
        let mut input = InputBuffer::new();
        for message in ["first", "second\nline two"] {
            input.insert_str(message);
            input.take();
        }
        input.insert_str("draft");

        input.up();
        assert_eq!(input.text(), "second\nline two");
        // Now on the last line of a two-line message: Up moves within it, then recalls.
        input.up();
        assert_eq!(input.text(), "second\nline two");
        input.up();
        assert_eq!(input.text(), "first");
        input.up();
        assert_eq!(input.text(), "first", "the oldest stays put");
        input.down();
        assert_eq!(input.text(), "second\nline two");
        input.down();
        assert_eq!(input.text(), "draft", "the draft comes back");
        input.down();
        assert_eq!(input.text(), "draft");
    }

    #[test]
    fn recalling_from_an_empty_input_works_and_empty_history_is_harmless() {
        let mut input = InputBuffer::new();
        input.up();
        assert!(input.is_empty());
        input.insert_str("hi");
        input.take();
        input.up();
        assert_eq!(input.text(), "hi");
    }

    #[test]
    fn repeated_messages_are_stored_once_in_a_row() {
        let mut input = InputBuffer::new();
        for _ in 0..3 {
            input.insert_str("same");
            input.take();
        }
        input.up();
        input.up();
        assert_eq!(input.text(), "same");
    }

    #[test]
    fn layout_wraps_and_places_the_cursor() {
        let input = typed("abcdefghij");
        let layout = input.layout(4);
        assert_eq!(layout.rows, vec!["abcd", "efgh", "ij"]);
        assert_eq!(layout.cursor, (2, 2));
    }

    #[test]
    fn a_cursor_after_a_full_row_moves_to_a_new_row() {
        let input = typed("abcd");
        let layout = input.layout(4);
        assert_eq!(layout.rows, vec!["abcd", ""]);
        assert_eq!(layout.cursor, (1, 0));
    }

    #[test]
    fn layout_counts_wide_characters_as_two_columns() {
        let input = typed("你好吗");
        let layout = input.layout(4);
        assert_eq!(layout.rows, vec!["你好", "吗"]);
        assert_eq!(layout.cursor, (1, 2));
    }

    #[test]
    fn layout_of_multiple_lines_and_an_empty_box() {
        assert_eq!(
            InputBuffer::new().layout(10),
            Layout {
                rows: vec![String::new()],
                cursor: (0, 0)
            }
        );
        let input = typed("a\n\nb");
        let layout = input.layout(10);
        assert_eq!(layout.rows, vec!["a", "", "b"]);
        assert_eq!(layout.cursor, (2, 1));
    }

    #[test]
    fn the_cursor_in_the_middle_of_a_wrapped_line() {
        let mut input = typed("abcdefgh");
        for _ in 0..5 {
            input.left();
        }
        assert_eq!(input.layout(4).cursor, (0, 3));
        input.right();
        assert_eq!(input.layout(4).cursor, (1, 0));
    }
}
