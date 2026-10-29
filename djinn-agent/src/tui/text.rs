//! Sanitising and wrapping of text from the model, tools and servers (design D6).
//!
//! Everything the TUI shows that did not come from the TUI itself goes through [`sanitise`] before
//! layout, so no layout or drawing code ever sees a raw terminal control character.

use ratatui::style::Style;
use ratatui::text::Span;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Replaces everything that could act as a terminal control sequence.
///
/// `\n` is kept (it is layout, not a control sequence), `\t` becomes four spaces, `\r` is dropped,
/// and every other C0/C1 control character becomes its visible Control Pictures glyph (`ESC` shows
/// as `␛`; `DEL` as `␡`; C1 controls as `�`).
pub fn sanitise(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '\n' => out.push('\n'),
            '\t' => out.push_str("    "),
            '\r' => {}
            '\u{7f}' => out.push('\u{2421}'),
            c if (c as u32) < 0x20 => {
                out.push(char::from_u32(0x2400 + c as u32).unwrap_or('\u{fffd}'))
            }
            c if ('\u{80}'..='\u{9f}').contains(&c) => out.push('\u{fffd}'),
            c => out.push(c),
        }
    }
    out
}

/// The number of terminal columns `text` takes.
pub fn width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// One line of styled pieces.
pub type Row = Vec<Span<'static>>;

/// The width of a row in columns.
pub fn row_width(row: &[Span<'_>]) -> usize {
    row.iter().map(|span| width(&span.content)).sum()
}

/// Appends `text` in `style`, merging with the previous piece when the style is the same.
fn push(row: &mut Row, text: &str, style: Style) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = row.last_mut()
        && last.style == style
    {
        last.content.to_mut().push_str(text);
        return;
    }
    row.push(Span::styled(text.to_owned(), style));
}

/// Wraps `text` (without newlines) at `width` columns, breaking inside characters' boundaries only.
/// Nothing is dropped; an empty text gives one empty row.
pub fn hard_wrap(text: &str, columns: usize, style: Style) -> Vec<Row> {
    let limit = columns.max(1);
    let mut rows = vec![Row::new()];
    let mut used = 0;
    let mut chunk = String::new();
    for c in text.chars() {
        let w = c.width().unwrap_or(0);
        if used > 0 && used + w > limit {
            push(rows.last_mut().expect("one row"), &chunk, style);
            chunk.clear();
            rows.push(Row::new());
            used = 0;
        }
        chunk.push(c);
        used += w;
    }
    push(rows.last_mut().expect("one row"), &chunk, style);
    rows
}

/// A run of non-space characters (possibly several styles) or a single space.
struct Token {
    pieces: Vec<(String, Style)>,
    width: usize,
    space: bool,
}

fn tokens(spans: &[Span<'static>]) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    for span in spans {
        for c in span.content.chars() {
            let space = c == ' ';
            let w = c.width().unwrap_or(0);
            match out.last_mut() {
                Some(last) if !space && !last.space => {
                    match last.pieces.last_mut() {
                        Some((text, style)) if *style == span.style => text.push(c),
                        _ => last.pieces.push((c.to_string(), span.style)),
                    }
                    last.width += w;
                }
                _ => out.push(Token {
                    pieces: vec![(c.to_string(), span.style)],
                    width: w,
                    space,
                }),
            }
        }
    }
    out
}

/// Greedy word wrap of styled text (which holds no newlines) to `columns`.
///
/// Spaces at the start of the text are kept, spaces at a break are dropped, and a word wider than a
/// row is split by columns. Always returns at least one row.
pub fn wrap_spans(spans: &[Span<'static>], columns: usize) -> Vec<Row> {
    let limit = columns.max(1);
    let mut rows: Vec<Row> = vec![Row::new()];
    let mut used = 0;
    let mut wrapped = false;
    let mut pending: Vec<&Token> = Vec::new();
    let tokens = tokens(spans);

    for token in &tokens {
        if token.space {
            pending.push(token);
            continue;
        }
        let gap: usize = pending.iter().map(|t| t.width).sum();
        if used > 0 && used + gap + token.width > limit {
            rows.push(Row::new());
            used = 0;
            wrapped = true;
        } else if used > 0 || !wrapped {
            for space in &pending {
                push(rows.last_mut().expect("one row"), " ", space.pieces[0].1);
            }
            used += gap;
        }
        pending.clear();

        if token.width > limit {
            // Too long for any row: split by columns.
            for (text, style) in &token.pieces {
                for c in text.chars() {
                    let w = c.width().unwrap_or(0);
                    if used > 0 && used + w > limit {
                        rows.push(Row::new());
                        used = 0;
                        wrapped = true;
                    }
                    push(rows.last_mut().expect("one row"), &c.to_string(), *style);
                    used += w;
                }
            }
        } else {
            for (text, style) in &token.pieces {
                push(rows.last_mut().expect("one row"), text, *style);
            }
            used += token.width;
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use ratatui::style::{Modifier, Style};

    use super::*;

    fn plain(rows: &[Row]) -> Vec<String> {
        rows.iter()
            .map(|row| row.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    fn wrap(text: &str, columns: usize) -> Vec<String> {
        plain(&wrap_spans(&[Span::raw(text.to_owned())], columns))
    }

    #[test]
    fn escape_and_other_controls_become_visible_glyphs() {
        assert_eq!(sanitise("a\u{1b}[2Jb"), "a\u{241b}[2Jb");
        assert_eq!(sanitise("\u{7}\u{0}\u{7f}"), "\u{2407}\u{2400}\u{2421}");
        assert_eq!(sanitise("\u{9b}31m"), "\u{fffd}31m");
    }

    #[test]
    fn newlines_stay_tabs_expand_and_carriage_returns_vanish() {
        assert_eq!(sanitise("a\r\nb\tc"), "a\nb    c");
    }

    #[test]
    fn ordinary_unicode_is_untouched() {
        let text = "héllo wörld 你好 🎉 — ünï";
        assert_eq!(sanitise(text), text);
    }

    #[test]
    fn short_text_stays_on_one_row() {
        assert_eq!(wrap("hello world", 20), vec!["hello world"]);
    }

    #[test]
    fn words_are_not_broken_and_break_spaces_are_dropped() {
        assert_eq!(
            wrap("the quick brown fox", 10),
            vec!["the quick", "brown fox"]
        );
        assert_eq!(wrap("aaaa bbbb", 4), vec!["aaaa", "bbbb"]);
    }

    #[test]
    fn a_word_longer_than_the_width_is_split_without_losing_characters() {
        let rows = wrap("abcdefghij", 4);
        assert_eq!(rows, vec!["abcd", "efgh", "ij"]);
        let rows = wrap("ab abcdefghij", 4);
        assert_eq!(rows, vec!["ab", "abcd", "efgh", "ij"]);
    }

    #[test]
    fn leading_spaces_on_the_first_row_are_kept() {
        assert_eq!(wrap("  indented", 20), vec!["  indented"]);
    }

    #[test]
    fn empty_text_is_one_empty_row() {
        assert_eq!(wrap("", 10), vec![String::new()]);
        assert_eq!(
            plain(&hard_wrap("", 10, Style::default())),
            vec![String::new()]
        );
    }

    #[test]
    fn width_one_never_loops_or_panics() {
        assert_eq!(wrap("ab cd", 1), vec!["a", "b", "c", "d"]);
        assert_eq!(wrap("你好", 1), vec!["你", "好"]);
    }

    #[test]
    fn wide_characters_count_two_columns() {
        assert_eq!(wrap("你好吗 你好", 6), vec!["你好吗", "你好"]);
        let rows = wrap("你好吗", 5);
        assert_eq!(rows, vec!["你好", "吗"]);
        for row in wrap_spans(&[Span::raw("🎉🎉🎉🎉")], 5) {
            assert!(row_width(&row) <= 5);
        }
    }

    #[test]
    fn styles_survive_wrapping_and_merge_when_equal() {
        let bold = Style::default().add_modifier(Modifier::BOLD);
        let spans = [Span::raw("plain "), Span::styled("bold words", bold)];
        let rows = wrap_spans(&spans, 8);
        assert_eq!(plain(&rows), vec!["plain", "bold", "words"]);
        assert_eq!(rows[1][0].style, bold);
        assert_eq!(rows[1].len(), 1);
    }

    #[test]
    fn a_word_made_of_several_styles_is_not_split_between_them() {
        let bold = Style::default().add_modifier(Modifier::BOLD);
        let spans = [Span::raw("x "), Span::styled("bo", bold), Span::raw("ld")];
        assert_eq!(plain(&wrap_spans(&spans, 4)), vec!["x", "bold"]);
    }

    #[test]
    fn hard_wrap_keeps_every_character_and_spaces() {
        let rows = hard_wrap("  let x = 1;", 5, Style::default());
        assert_eq!(plain(&rows), vec!["  let", " x = ", "1;"]);
        assert_eq!(plain(&rows).concat(), "  let x = 1;");
    }

    #[test]
    fn rows_never_exceed_the_width() {
        let text =
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod 你好 🎉";
        for columns in 1..40 {
            for row in wrap_spans(&[Span::raw(text)], columns) {
                let w = row_width(&row);
                // A single double-width character can exceed a width of one.
                assert!(w <= columns.max(2), "{columns}: {w}");
            }
        }
    }
}
