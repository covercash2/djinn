//! Which interactive front end to start.

/// How the interactive session is presented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontEnd {
    /// The full-screen terminal UI.
    Tui,
    /// The line-based REPL, which also works through pipes.
    Plain,
}

/// The TUI needs a terminal on both ends and the `tui` feature; `--plain` always wins.
pub fn choose(
    plain: bool,
    stdin_is_tty: bool,
    stdout_is_tty: bool,
    tui_compiled: bool,
) -> FrontEnd {
    if !plain && tui_compiled && stdin_is_tty && stdout_is_tty {
        FrontEnd::Tui
    } else {
        FrontEnd::Plain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_terminal_gets_the_tui() {
        assert_eq!(choose(false, true, true, true), FrontEnd::Tui);
    }

    #[test]
    fn a_pipe_on_either_end_gets_the_plain_repl() {
        assert_eq!(choose(false, false, true, true), FrontEnd::Plain);
        assert_eq!(choose(false, true, false, true), FrontEnd::Plain);
        assert_eq!(choose(false, false, false, true), FrontEnd::Plain);
    }

    #[test]
    fn plain_forces_the_repl_on_a_terminal() {
        assert_eq!(choose(true, true, true, true), FrontEnd::Plain);
    }

    #[test]
    fn without_the_feature_a_terminal_gets_the_repl() {
        assert_eq!(choose(false, true, true, false), FrontEnd::Plain);
    }
}
