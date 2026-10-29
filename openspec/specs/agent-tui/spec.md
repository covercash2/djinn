# agent-tui Specification

## Purpose
Defines the full-screen terminal front end of the agent: a scrollable conversation with
Markdown-rendered replies, tool calls shown as expandable blocks, approval prompts before
tools run, multi-line input with history, a status bar and slash commands. It must be safe
to show untrusted model text, always restore the terminal, and be testable without a
terminal or a model server.

## Requirements

### Requirement: Full-screen conversation view
The TUI SHALL show the conversation in a scrollable transcript: each user message,
the assistant's reply (shown incrementally while it streams), and each tool call.
The transcript SHALL be scrollable with the keyboard (page up/down; ctrl-home/ctrl-end jump to the
start and the end) and the mouse wheel.

#### Scenario: Streamed answer
- **WHEN** the model streams "Hello" and then " world"
- **THEN** the transcript shows "Hello" and then "Hello world" in one assistant message without waiting for the turn to end

#### Scenario: Following the output
- **WHEN** new content arrives and the view is at the bottom
- **THEN** the view stays at the bottom

#### Scenario: Reading history while the model writes
- **WHEN** the user scrolls up during a streaming reply
- **THEN** the view stays on the lines being read and an indicator shows that there is newer content below

### Requirement: Markdown rendering
Assistant text SHALL be rendered as Markdown: headings, bold, italic, inline code,
fenced code blocks, bullet and numbered lists, block quotes and horizontal rules.
Text SHALL wrap to the width of the view without breaking words, and SHALL stay
correct when the terminal is resized. Code lines SHALL never be word-wrapped or
merged with surrounding text; a code line wider than the view continues on the next
row. Text that is not Markdown SHALL be shown unchanged.

#### Scenario: Code fence
- **WHEN** the reply contains a fenced code block with a line longer than the view
- **THEN** the code lines are shown in a distinct style, none of their characters are lost, and they are not joined with the surrounding text

#### Scenario: Plain text
- **WHEN** the reply contains no Markdown syntax
- **THEN** it is shown as written, only wrapped

### Requirement: Partial and hostile text is safe to show
Incomplete Markdown that arrives mid-stream (an unclosed code fence or emphasis marker)
SHALL render as readable text and SHALL NOT hide or reorder the text that follows. Terminal
control characters in model, tool or server text SHALL NOT reach the terminal as control
sequences.

#### Scenario: Unclosed fence mid-stream
- **WHEN** the stream has delivered "```rust\nfn main() {" and nothing more
- **THEN** the partial code is shown as a code block and the final render, once the fence closes, is identical to rendering the complete text at once

#### Scenario: Escape sequences are neutralised
- **WHEN** a tool result contains the bytes `ESC [ 2 J`
- **THEN** the screen is not cleared; the characters are shown as visible placeholders

### Requirement: Tool call blocks
Each tool call SHALL appear in the transcript as a block showing the tool name and
its arguments, its state (waiting for approval, running, succeeded, failed or
denied), and, once finished, its duration and result. A block SHALL be collapsed to
one summary line by default and expandable by the user; results longer than the view
SHALL be shown truncated with a marker until expanded.

#### Scenario: Successful call
- **WHEN** the model calls `calculator` with `2+2` and it returns `4`
- **THEN** the block shows `calculator`, the arguments, a success state and the duration, and expanding it shows `4`

#### Scenario: Failed call
- **WHEN** a tool returns an error
- **THEN** its block is marked failed and the error text is visible when expanded

#### Scenario: Denied call
- **WHEN** the user denies a call
- **THEN** its block is marked denied and shows the reason, if one was given

### Requirement: Approval prompt
When approval is required for a call (see `agent-loop`), the TUI SHALL show the tool
name and full arguments and wait for the user to choose. The agent SHALL NOT continue,
and the tool SHALL NOT run, until the user has chosen. While waiting the user SHALL
still be able to cancel the turn and scroll the transcript. In `ask` mode every call
is confirmed; in `auto` mode only calls of tools that declare they need confirmation
(see `agent-tools`) are.

#### Scenario: Ask mode
- **WHEN** the TUI runs with `--approval ask` and the model calls `calculator`
- **THEN** the call waits for the user's decision before it runs

#### Scenario: Auto mode
- **WHEN** the TUI runs with `--approval auto` and the model calls `calculator`
- **THEN** the tool runs without a prompt

#### Scenario: Cancel while waiting
- **WHEN** the user presses Ctrl-C while an approval is pending
- **THEN** the turn is cancelled, the tool does not run, and the conversation history is as it was before the turn

### Requirement: Approval choices
The user SHALL be able to allow the call once, allow that tool for the rest of the
session, or deny it with an optional reason. "Allow for the session" SHALL NOT be
offered for a tool that declares it needs confirmation; that tool is asked every time.
The session allowance ends when the conversation is reset or the program ends.

#### Scenario: Allow once
- **WHEN** the user allows a call
- **THEN** the tool runs, and the next call of the same tool asks again

#### Scenario: Allow for the session
- **WHEN** the user chooses "allow for this session" for `read_file`
- **THEN** later `read_file` calls run without asking until the conversation is reset or the program ends, while other tools still ask

#### Scenario: Deny
- **WHEN** the user denies a call and types the reason "not that file"
- **THEN** the tool does not run and the model receives a tool result saying the user denied the call and why

### Requirement: Input editing
The TUI SHALL provide a multi-line input box. Alt-Enter (or Shift-Enter where the
terminal reports it) SHALL insert a newline. The input SHALL support cursor movement
and editing, and Up/Down at the first/last line SHALL recall earlier messages of the
session. Pasted text SHALL be inserted as typed text, not executed as key presses.

#### Scenario: Multi-line paste
- **WHEN** the user pastes three lines of text
- **THEN** the input box holds the three lines and nothing is sent until Enter is pressed

#### Scenario: History
- **WHEN** the user presses Up on the first line of an empty input after sending a message
- **THEN** that message is placed in the input for editing

### Requirement: Sending messages
Enter SHALL send the message. A blank message SHALL NOT be sent. While a turn is
running the user SHALL be able to type the next message but not send it until the turn
ends.

#### Scenario: Blank message
- **WHEN** the user presses Enter on an empty or whitespace-only input
- **THEN** no model request is made

#### Scenario: Typing while busy
- **WHEN** a turn is running and the user types and presses Enter
- **THEN** the text stays in the input and is not sent until the turn has ended

### Requirement: Status bar
A status bar SHALL show the model name, the server's base URL (without credentials),
the approval mode, whether the agent is idle or working (with the model-call count
of the current turn against the step limit), and the key hints relevant to the current
state.

#### Scenario: Status while working
- **WHEN** the second model call of a turn with a limit of 8 is running
- **THEN** the status bar shows a working state and `2/8`

### Requirement: Slash commands
Lines starting with `/` SHALL be commands, the same ones as in the plain REPL: `/exit`
and `/quit` end the session, `/reset` clears the conversation and the transcript,
`/tools` lists the tools with their descriptions, `/help` lists the commands and keys.
An unknown command SHALL show an error in the transcript and send nothing to the model.

#### Scenario: Unknown command
- **WHEN** the user enters `/frobnicate`
- **THEN** an error and the list of commands appear in the transcript and no request is made

#### Scenario: Reset
- **WHEN** the user enters `/reset`
- **THEN** the transcript is cleared and the next model request contains only the system message and the new message

### Requirement: Interrupt and quitting
Ctrl-C while a turn is running SHALL cancel it (including an in-flight model request
or tool), discard the turn from the conversation history, and show that it was
cancelled. Ctrl-C while idle SHALL NOT quit; it SHALL show how to quit. `/exit`, `/quit`
and Ctrl-D on an empty input SHALL end the program with exit code 0.

#### Scenario: Cancel a long answer
- **WHEN** the user presses Ctrl-C while an answer is streaming
- **THEN** streaming stops, the transcript marks the turn cancelled, and the next message is answered without the cancelled turn in its history

#### Scenario: Ctrl-C when idle
- **WHEN** the user presses Ctrl-C with no turn running
- **THEN** the program keeps running and a hint explains `/exit` and Ctrl-D

### Requirement: Errors are shown, not fatal
A backend error, the step limit, or an unreachable server SHALL be shown in the
transcript with the same message the REPL would print, and the session SHALL remain
usable. At start-up the TUI SHALL warn (in the transcript) if the server cannot be
reached, but still start.

#### Scenario: Server down at start
- **WHEN** the TUI starts while the server is down and the server is started afterwards
- **THEN** the next message succeeds without restarting the TUI

#### Scenario: Step limit
- **WHEN** a turn reaches the step limit
- **THEN** the transcript shows the step-limit message and the next message is accepted

### Requirement: Terminal state is always restored
The TUI SHALL switch to the alternate screen and raw mode only while it runs and
SHALL restore the terminal on normal exit, on `/exit`, on an error return and on a
panic (the panic message SHALL then be printed to the restored terminal). It SHALL
adapt to terminal resizes without losing content.

#### Scenario: Panic
- **WHEN** the TUI code panics
- **THEN** the terminal is left in its normal mode (cooked, main screen, cursor visible) and the panic message is readable

#### Scenario: Resize
- **WHEN** the terminal is resized while a reply is streaming
- **THEN** the transcript is re-wrapped to the new width and no text is lost

### Requirement: Testable without a terminal or server
Given a scripted backend and synthetic key events the TUI SHALL render deterministic
screens whose content can be checked, so its behaviour is verifiable in automated tests.

#### Scenario: Scripted session
- **WHEN** a test drives the TUI with a scripted backend and the keys "2+2", Enter
- **THEN** the rendered screen contains the user message, a tool block for `calculator`, and the scripted answer

### Requirement: Slash command completion
While the user is typing a command name in the TUI input (the input is a single line
that starts with `/` and the cursor is still within the first word), the TUI SHALL show
a popup above the input box listing the commands whose name or alias starts with the
typed text, ignoring case, in the order of `/help`, each with a one-line description.
Typing `/` alone SHALL list every command. The popup SHALL be shown only when the agent
is idle, and SHALL NOT be shown when nothing matches.

#### Scenario: Popup on slash
- **WHEN** the user types `/` into an empty input
- **THEN** a popup lists `/exit`, `/quit`, `/reset`, `/tools` and `/help` with descriptions

#### Scenario: Narrowing
- **WHEN** the user has typed `/re`
- **THEN** the popup lists only `/reset`

#### Scenario: Aliases match
- **WHEN** the user has typed `/qu`
- **THEN** the popup lists `/quit`

#### Scenario: No match
- **WHEN** the user has typed `/zzz`
- **THEN** no popup is shown, and pressing Enter reports the unknown command exactly as before

#### Scenario: Not a command
- **WHEN** the input starts with `/` but has a second line, or the cursor is after the first space, or the agent is working
- **THEN** no popup is shown

### Requirement: Choosing a completion
While the popup is open, Up and Down SHALL move the selection (wrapping at both ends)
instead of recalling earlier messages, Tab SHALL replace the typed command name with
the selected command, and Esc SHALL close the popup until the input changes. The first
row SHALL be selected whenever the list changes.

#### Scenario: Accept with Tab
- **WHEN** the user has typed `/re` and presses Tab
- **THEN** the input becomes `/reset`, nothing is sent, and the popup now shows only the exact match

#### Scenario: Navigate
- **WHEN** the popup shows `/exit`, `/quit` and the user presses Down once and then Tab
- **THEN** the input becomes `/quit`

#### Scenario: Dismiss
- **WHEN** the user presses Esc with the popup open and then types another character that keeps matching
- **THEN** the popup is closed after Esc and appears again with the new character

#### Scenario: History still works without the popup
- **WHEN** the popup is closed and the input is empty
- **THEN** Up recalls the previous message as before

### Requirement: Enter with the popup open
Enter SHALL accept the selected command when it differs from what was typed, without
running it; when the typed name already equals the selected command, Enter SHALL run it
as it would without the popup.

#### Scenario: Enter completes first, then runs
- **WHEN** the user has typed `/to` and presses Enter twice
- **THEN** the first Enter turns the input into `/tools` without running it, and the second runs `/tools`
