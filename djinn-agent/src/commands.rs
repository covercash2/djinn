//! Slash commands, shared by the plain REPL and the TUI so the two cannot drift apart.

/// A `/command` typed instead of a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// `/exit` or `/quit`: end the session.
    Exit,
    /// `/reset`: forget the conversation.
    Reset,
    /// `/tools`: list the available tools.
    Tools,
    /// `/help`: list the commands.
    Help,
}

/// The text after the `/` named no command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownCommand(pub String);

/// The commands, one per line, as shown by `/help`.
pub const HELP: &str = "/exit (or /quit)  end the session\n/reset            forget the conversation\n/tools            list the available tools\n/help             show this list";

impl Command {
    /// Parses the text after the leading `/` (surrounding whitespace is ignored).
    pub fn parse(text: &str) -> Result<Self, UnknownCommand> {
        match text.trim() {
            "exit" | "quit" => Ok(Self::Exit),
            "reset" => Ok(Self::Reset),
            "tools" => Ok(Self::Tools),
            "help" => Ok(Self::Help),
            other => Err(UnknownCommand(other.to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_command_and_alias_parses() {
        assert_eq!(Command::parse("exit"), Ok(Command::Exit));
        assert_eq!(Command::parse("quit"), Ok(Command::Exit));
        assert_eq!(Command::parse(" reset "), Ok(Command::Reset));
        assert_eq!(Command::parse("tools"), Ok(Command::Tools));
        assert_eq!(Command::parse("help"), Ok(Command::Help));
    }

    #[test]
    fn unknown_commands_keep_their_text() {
        assert_eq!(
            Command::parse("frobnicate now"),
            Err(UnknownCommand("frobnicate now".into()))
        );
        assert_eq!(Command::parse(""), Err(UnknownCommand(String::new())));
    }

    #[test]
    fn the_help_text_lists_every_command() {
        for name in ["/exit", "/quit", "/reset", "/tools", "/help"] {
            assert!(HELP.contains(name), "{name}");
        }
    }
}
