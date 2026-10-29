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

/// One name a command can be typed as. A command with an alias has one row per name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandInfo {
    /// What is typed after the `/`.
    pub name: &'static str,
    pub command: Command,
    /// A short description, as in `/help` and the completion popup.
    pub description: &'static str,
}

/// Every name, in the order `/help` lists them. The single source for parsing and completion.
pub const COMMANDS: &[CommandInfo] = &[
    CommandInfo {
        name: "exit",
        command: Command::Exit,
        description: "end the session",
    },
    CommandInfo {
        name: "quit",
        command: Command::Exit,
        description: "end the session",
    },
    CommandInfo {
        name: "reset",
        command: Command::Reset,
        description: "forget the conversation",
    },
    CommandInfo {
        name: "tools",
        command: Command::Tools,
        description: "list the available tools",
    },
    CommandInfo {
        name: "help",
        command: Command::Help,
        description: "show this list",
    },
];

impl Command {
    /// Parses the text after the leading `/` (surrounding whitespace is ignored).
    pub fn parse(text: &str) -> Result<Self, UnknownCommand> {
        let text = text.trim();
        COMMANDS
            .iter()
            .find(|info| info.name == text)
            .map(|info| info.command)
            .ok_or_else(|| UnknownCommand(text.to_owned()))
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
    fn the_table_has_unique_names_and_every_row_parses_to_its_command() {
        let mut names: Vec<_> = COMMANDS.iter().map(|info| info.name).collect();
        for info in COMMANDS {
            assert_eq!(Command::parse(info.name), Ok(info.command), "{}", info.name);
        }
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), COMMANDS.len());
    }

    #[test]
    fn the_help_text_stays_in_step_with_the_table() {
        for info in COMMANDS {
            assert!(HELP.contains(&format!("/{}", info.name)), "{}", info.name);
            assert!(HELP.contains(info.description), "{}", info.description);
        }
        // The names appear in table order.
        let positions: Vec<_> = COMMANDS
            .iter()
            .map(|info| HELP.find(&format!("/{}", info.name)).unwrap())
            .collect();
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "{positions:?}"
        );
    }

    #[test]
    fn the_help_text_lists_every_command() {
        for name in ["/exit", "/quit", "/reset", "/tools", "/help"] {
            assert!(HELP.contains(name), "{name}");
        }
    }
}
