//! Completion of slash commands while they are typed (design D2).
//!
//! A pure function of the input text and the cursor: nothing here is remembered, so there is no
//! stale state to invalidate when the input changes.

use crate::commands::{COMMANDS, CommandInfo};

/// The commands that fit what has been typed after the `/`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completions {
    /// The typed text between the `/` and the cursor.
    pub prefix: String,
    /// Matching commands, in the order of `/help`. Never empty.
    pub rows: Vec<&'static CommandInfo>,
}

/// The completions for `text` with the cursor at (`row`, `col`) (`col` counted in characters), or
/// `None` when the input is not a command being typed: it must be one line that starts with `/`, with
/// the cursor after the `/` and still inside the first word, and at least one name must match.
pub fn completions(text: &str, row: usize, col: usize) -> Option<Completions> {
    if row != 0 || col == 0 || text.contains('\n') {
        return None;
    }
    let rest = text.strip_prefix('/')?;
    let typed: String = rest.chars().take(col - 1).collect();
    if typed.chars().count() != col - 1 || typed.chars().any(char::is_whitespace) {
        return None; // the cursor is beyond the end of the text, or after the first word
    }
    let wanted = typed.to_ascii_lowercase();
    let rows: Vec<_> = COMMANDS
        .iter()
        .filter(|info| info.name.starts_with(&wanted))
        .collect();
    (!rows.is_empty()).then_some(Completions {
        prefix: typed,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(text: &str) -> Option<Vec<&'static str>> {
        let col = text.chars().count();
        completions(text, 0, col).map(|c| c.rows.iter().map(|r| r.name).collect())
    }

    #[test]
    fn a_slash_alone_lists_every_command_in_help_order() {
        assert_eq!(
            names("/"),
            Some(vec!["exit", "quit", "reset", "tools", "help"])
        );
    }

    #[test]
    fn typing_narrows_the_list() {
        assert_eq!(names("/re"), Some(vec!["reset"]));
        assert_eq!(names("/e"), Some(vec!["exit"]));
        assert_eq!(names("/reset"), Some(vec!["reset"]));
        assert_eq!(names("/h"), Some(vec!["help"]));
    }

    #[test]
    fn aliases_match_like_any_other_name() {
        assert_eq!(names("/qu"), Some(vec!["quit"]));
    }

    #[test]
    fn matching_ignores_case() {
        assert_eq!(names("/RE"), Some(vec!["reset"]));
        assert_eq!(names("/ToOl"), Some(vec!["tools"]));
    }

    #[test]
    fn no_match_means_no_completions() {
        assert_eq!(names("/zzz"), None);
        assert_eq!(names("/resetx"), None);
    }

    #[test]
    fn only_a_single_line_starting_with_a_slash_qualifies() {
        assert_eq!(names(""), None);
        assert_eq!(names("hello"), None);
        assert_eq!(
            names(" /re"),
            None,
            "a leading space is a message, not a command"
        );
        assert_eq!(completions("/re\nmore", 0, 3), None);
        assert_eq!(
            completions("/re", 1, 3),
            None,
            "the cursor is on another line"
        );
    }

    #[test]
    fn the_cursor_must_be_inside_the_first_word() {
        assert_eq!(completions("/re now", 0, 7), None, "after the space");
        assert_eq!(completions("/re now", 0, 4), None, "right after the space");
        assert_eq!(completions("/reset", 0, 0), None, "before the slash");
        // Inside the word, only the text before the cursor counts.
        let inside = completions("/reset", 0, 3).unwrap();
        assert_eq!(inside.prefix, "re");
        assert_eq!(inside.rows.len(), 1);
        assert_eq!(
            completions("/re", 0, 9),
            None,
            "a cursor past the end is not real"
        );
    }

    #[test]
    fn the_prefix_is_what_was_typed() {
        assert_eq!(completions("/To", 0, 3).unwrap().prefix, "To");
        assert_eq!(completions("/", 0, 1).unwrap().prefix, "");
    }
}
