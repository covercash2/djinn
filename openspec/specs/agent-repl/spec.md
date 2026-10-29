# agent-repl Specification

## Purpose
Defines the interactive command-line front end of the agent: a plain line-based
REPL (no full-screen UI yet) plus a one-shot mode for scripting and testing.

## Requirements

### Requirement: Interactive session
Running `djinn-agent` with no prompt argument SHALL start a read-eval-print loop
that shows a prompt, reads one line as a user message, runs a turn, prints the
answer, and repeats until the user exits. The assistant's text SHALL be printed as
it streams. Blank lines SHALL be ignored.

#### Scenario: Question and streamed answer
- **WHEN** the user types a question
- **THEN** the answer appears incrementally on standard output and a new prompt is shown after it ends

#### Scenario: Blank input
- **WHEN** the user presses enter on an empty line
- **THEN** no model request is made and the prompt is shown again

### Requirement: Tool activity is visible
When the agent calls a tool the REPL SHALL show the tool name and arguments before
it runs and a one-line summary of the outcome after it finishes. This information
SHALL be written to standard error so standard output contains only the answer text.

#### Scenario: Tool call shown
- **WHEN** the model calls `calculator` with `2+2`
- **THEN** standard error shows the call and its result `4`, and standard output shows only the final answer

### Requirement: Slash commands
Lines starting with `/` SHALL be treated as commands, not messages: `/exit` (and
`/quit`) ends the session, `/reset` clears the conversation, `/tools` lists the
registered tools with descriptions, and `/help` lists the commands. An unknown
command SHALL print an error and the list of commands without sending anything to
the model.

#### Scenario: Exit
- **WHEN** the user enters `/exit` or sends end-of-file (Ctrl-D)
- **THEN** the program ends with exit code 0

#### Scenario: Unknown command
- **WHEN** the user enters `/frobnicate`
- **THEN** an error and the command list are shown and no model request is made

#### Scenario: List tools
- **WHEN** the user enters `/tools`
- **THEN** each registered tool is listed with its name and description

### Requirement: Interrupt
Pressing Ctrl-C during a turn SHALL cancel the turn (including an in-flight model
request), discard that turn from the history, and return to the prompt. Pressing
Ctrl-C at an empty prompt SHALL NOT exit the program.

#### Scenario: Cancel a long answer
- **WHEN** the user presses Ctrl-C while an answer is streaming
- **THEN** streaming stops, the prompt returns, and the cancelled turn is not in the conversation history

### Requirement: Configuration
The program SHALL accept command-line flags, each with an environment variable
fallback, for: base URL (`--base-url`, `DJINN_AGENT_BASE_URL`), model name
(`--model`, `DJINN_AGENT_MODEL`), API key (`--api-key`, `DJINN_AGENT_API_KEY`),
step limit (`--max-steps`), file-tool root directory (`--tool-root`), and a file
containing the system prompt (`--system`). Command-line flags SHALL override
environment variables. Invalid values SHALL produce a usage error and exit code 2.

#### Scenario: Flag overrides environment
- **WHEN** `DJINN_AGENT_MODEL=a` is set and the program is run with `--model b`
- **THEN** requests use model `b`

#### Scenario: Custom system prompt
- **WHEN** `--system prompts/system/help_chris.txt` is given
- **THEN** the conversation starts with that file's contents as the system message

#### Scenario: Missing system prompt file
- **WHEN** `--system` names a file that does not exist
- **THEN** the program prints an error naming the file and exits with a non-zero code before contacting the server

### Requirement: One-shot mode
`djinn-agent --once "<text>"` SHALL run a single turn with that text, print only
the answer to standard output, and exit with code 0, or non-zero when the turn
failed (backend error or step limit).

#### Scenario: Script usage
- **WHEN** `djinn-agent --once "What is 2+2? Use the calculator."` runs against a working backend
- **THEN** the answer is on standard output, tool activity is on standard error, and the exit code is 0

#### Scenario: Backend unreachable
- **WHEN** the backend cannot be reached
- **THEN** an error naming the URL is on standard error, standard output is empty, and the exit code is non-zero

### Requirement: Startup check
Before the first prompt the program SHOULD verify the backend is reachable. If it
is not, it SHALL print a warning with the URL but still start, so the user can fix
the server and retry without restarting.

#### Scenario: Backend started later
- **WHEN** the REPL starts while the server is down and the server is started afterwards
- **THEN** the next user message succeeds without restarting the REPL

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
