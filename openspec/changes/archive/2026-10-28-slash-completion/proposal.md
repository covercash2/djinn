# Proposal

> **Sequence:** builds on the archived `agent-tui` change. Touches only `djinn-agent`
> (`commands.rs` and the `tui` module); no wire format, dependency or CLI flag changes.

## Why

The slash commands (`/exit`, `/quit`, `/reset`, `/tools`, `/help`) can only be
discovered by typing `/help` or reading the README, and a typo costs a round trip
through an "unknown command" notice. A popup that appears as soon as `/` is typed and
narrows as the user types makes them discoverable and quick to enter, and it stays
correct as commands are added because it reads the same command list as `/help`.

## What Changes

- **Completion popup in the TUI.** While the user is typing the command name (the input
  is one line, starts with `/`, and the cursor is still inside the first word), a popup
  above the input box lists the commands whose name starts with what was typed, each
  with a one-line description. The selected row is highlighted.
- **Keys.** Up/Down move the selection (wrapping) instead of recalling history while the
  popup is open; Tab accepts the selection; Enter accepts it when it differs from what
  was typed and otherwise runs the command as before; Esc closes the popup until the
  input changes. No match means no popup, and Enter behaves exactly as today (unknown
  command notice).
- **One command table.** `commands.rs` gets a table of commands (name, aliases,
  description) that is the single source for parsing, `/help` and the completions, with
  a test that keeps the hand-written help text in step with it.
- Plain REPL unchanged (no popup, same commands, same output).

Out of scope: fuzzy matching, completion of command arguments (no command takes any),
completion in the plain REPL, mouse selection in the popup, user-defined commands.

## Capabilities

### New Capabilities

_None._

### Modified Capabilities

- `agent-tui`: adds the requirement "Slash command completion".

## Impact

- `djinn-agent/src/commands.rs`: command table.
- `djinn-agent/src/tui/{app,view}.rs`: completion state, key handling, popup drawing;
  snapshot tests.
- README keys table and `/help` text mention the popup. No new dependencies; no
  `djinn-core` change; no new free-form string settings (candidates are values of the
  existing `Command` enum).
