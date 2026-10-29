# Proposal

> **Sequence:** builds on the archived `add-agent-harness` and `typed-domain-values`
> changes (both implemented). It touches only `djinn-agent` (plus the justfile/README);
> no wire format and no other crate changes.

## Why

The agent's only interactive front end is a line-based REPL: the answer streams into
the same scrollback as tool logs, there is no way to review a long conversation, a
tool call is shown as one cramped line, and nothing can stop a tool before it runs
(today that is fine because every built-in tool is read-only, but the first tool that
writes or runs something needs a confirmation step). A full-screen terminal UI makes
the agent loop *visible* and *controllable*, which is what the next round of agent
work (more tools, longer sessions) needs.

## What Changes

- **New `ratatui` front end** inside `djinn-agent`, behind a default-on `tui` cargo
  feature (so `--no-default-features` keeps the crate light):
  - scrollable transcript of user messages, streamed assistant text rendered as
    Markdown (headings, emphasis, lists, quotes, inline code, fenced code blocks);
  - tool calls shown as blocks (name, arguments, status running/ok/failed, duration,
    result) that can be expanded or collapsed;
  - multi-line input box with history, a status bar (model, server, step count,
    busy/idle, approval mode) and slash commands (`/exit`, `/quit`, `/reset`,
    `/tools`, `/help`, same as the REPL);
  - Ctrl-C cancels the running turn with the existing semantics (turn discarded from
    history, session stays usable); at an idle prompt it does not quit;
  - the terminal is always restored, also on panic and on `/exit`.
- **Tool approval.** The agent loop gets an approval step before each tool runs. In
  the TUI a call can be allowed once, allowed for the rest of the session, or denied
  (optionally with a reason that is reported to the model as the tool result). A
  `--approval` flag selects `ask` (default in the TUI: every call is confirmed) or
  `auto` (only tools that declare they need confirmation are confirmed). The plain
  REPL and `--once` keep running tools without asking, as today.
- **Front end selection.** `djinn-agent` with no `--once` starts the TUI when both
  stdin and stdout are terminals, and the plain REPL otherwise; `--plain` forces the
  REPL. Piped usage and the existing binary tests are therefore unchanged.
- A `Tool` can declare that it needs confirmation (default: no). The built-in tools
  remain read-only and do not.
- Docs: README section and `just agent` keep working; `CLAUDE.md` mentions the TUI.

Out of scope: mouse text selection/copy handling beyond what the terminal does, themes
and key remapping, persistent session history on disk, new tools, changes to the
backend or wire types, and a TUI in any other crate.

## Capabilities

### New Capabilities

- `agent-tui`: the full-screen terminal front end — transcript, Markdown rendering,
  tool blocks, input, status bar, slash commands, interrupt, terminal restoration.

### Modified Capabilities

- `agent-repl`: selecting between the TUI and the plain REPL (`--plain`, terminal
  detection); the REPL and `--once` behave as before.
- `agent-loop`: a tool call can be approved or denied by the user before it runs.
- `agent-tools`: a tool can declare that it needs user confirmation.

## Impact

- `djinn-agent`: new `tui` module tree, new dependencies `ratatui`, `crossterm`
  (event stream), `pulldown-cmark`, `unicode-width`; `Agent::run_turn` gains an
  approval-aware sibling (the existing method keeps its signature); `Tool` gains a
  defaulted method; CLI gains `--plain` and `--approval`.
- Workspace `Cargo.toml`: the new dependencies in `[workspace.dependencies]`.
- `just check` must stay green, including `check-agent-deps` (ratatui/crossterm are
  allowed; candle, `djinn-models` and `djinn-server` are still forbidden).
- No change to `djinn-core`, `djinn-models`, `djinn-server`, `djinn-cli` or any
  serialized format. No new `String` domain values: the new settings are enums
  (`Approval`, `FrontEnd`) and typed values already in `djinn_core::values`.
