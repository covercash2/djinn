# agent-repl Specification

## ADDED Requirements

### Requirement: Front end selection
When `--once` is not given, `djinn-agent` SHALL start the full-screen TUI (see
`agent-tui`) if both standard input and standard output are terminals, and the plain
line-based REPL otherwise. `--plain` SHALL force the plain REPL even on a terminal.
The plain REPL and `--once` SHALL behave exactly as specified in this capability
regardless of whether the TUI is compiled in.

#### Scenario: Piped input
- **WHEN** `djinn-agent` is run with standard input from a pipe
- **THEN** the plain REPL runs and no terminal control sequences are written

#### Scenario: Interactive terminal
- **WHEN** `djinn-agent` is run in a terminal without `--once` or `--plain`
- **THEN** the TUI starts

#### Scenario: Forced plain mode
- **WHEN** `djinn-agent --plain` is run in a terminal
- **THEN** the line-based REPL starts

#### Scenario: TUI not compiled in
- **WHEN** the binary was built without the `tui` feature and is run in a terminal
- **THEN** the plain REPL starts

### Requirement: Shared configuration
The same flags and environment variables SHALL configure both front ends.
`--approval` (`ask` or `auto`, default `ask`) applies to the TUI only; it SHALL be
validated at start-up like every other flag, and accepted and ignored with `--plain`
and `--once`.

#### Scenario: Approval flag with the plain REPL
- **WHEN** `djinn-agent --plain --approval auto` is started
- **THEN** it starts normally and tools run without asking

#### Scenario: Invalid approval mode
- **WHEN** `djinn-agent --approval sometimes` is run
- **THEN** it exits with code 2 and names `--approval` and the valid values
