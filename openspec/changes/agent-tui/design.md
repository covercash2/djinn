# Design

See `proposal.md` for why and what; the requirements are in `specs/`. Everything
below is inside the `djinn-agent` crate unless stated.

## Context

- `Agent::run_turn(&mut self, message, FnMut(AgentEvent)) -> Result<String, TurnError>`
  already streams `AgentEvent`s (`Text`, `ToolStarted`, `ToolFinished`, `StepLimit`,
  `TurnFinished`) and leaves the history untouched when the future is dropped, which is
  how the REPL implements Ctrl-C. The TUI reuses this; it does not need a new loop.
- `repl::run` drives stdin lines and a `Notify` for SIGINT; its tests and the 13 binary
  tests in `tests/cli.rs` pipe stdin, so they exercise the plain path only.
- `djinn-agent` must not depend on candle, `djinn-models` or `djinn-server`
  (`just check-agent-deps`). Terminal crates are fine.
- ratatui 0.30 and crossterm 0.29 are the current releases.

## Goals / Non-Goals

**Goals:** a TUI whose behaviour is testable without a terminal; the agent loop stays
the single source of truth for turns; approval that cannot be bypassed by a UI bug
(the tool is only reachable through the loop, which asks first); no regression for
piped use.

**Non-goals:** on-disk session history, themes/keymaps, a widget library, mouse text
selection, running tools in parallel, TUI for other binaries.

## Decisions

### D1. Same crate, `tui` feature, module tree
`djinn-agent/src/tui/` with `mod.rs` (entry point `run`), `app.rs` (state + update),
`view.rs` (rendering), `markdown.rs`, `text.rs` (sanitise + wrap), `input.rs`,
`approval.rs`, `terminal.rs` (guard). The feature `tui` (default on) gates the module
and the dependencies (`ratatui`, `crossterm`, `pulldown-cmark`, `unicode-width`, all optional).
*Alternative:* a `djinn-tui` crate — rejected (user choice): it would duplicate the
CLI/config wiring for no isolation gain, since the agent crate stays free of the
inference stack either way.

### D2. Model-update-view with effects; a single task, no spawning
`App` is plain data. `App::update(Msg) -> Vec<Effect>` is pure and synchronous;
`view::render(&App, &mut Frame)` is a pure function of the state (it also records the
layout facts the next `update` needs, e.g. viewport height for page-up). `Msg` is
`Key`, `Paste`, `Mouse`, `Resize`, `Tick`, `Agent(AgentEvent)`,
`Approval(PendingApproval)`, `TurnEnded(Result<String, TurnError>)`. `Effect` is
`Send(String)`, `Cancel`, `Reset`, `Quit`; the runtime performs them.

The runtime (`tui::run`) is one `async fn` with one `tokio::select!` over: the crossterm
`EventStream`, an `mpsc::UnboundedReceiver<Msg>` fed by the turn, a 100 ms ticker that
is only polled while a turn is running (spinner), and the turn future itself
(`Option<Pin<Box<dyn Future>>>`, `None` when idle). The turn future borrows
`&mut Agent` and the sender; nothing is `spawn`ed, so there are no `Send`/`'static`
requirements on `Backend` or tools and cancelling is *dropping the future* (the
existing, tested semantics). *Alternative:* run the agent in a spawned task and talk
over channels — rejected: it needs `Agent: 'static`, and cancellation would need an
extra protocol (abort handles, history consistency).

### D2a. Progress events
`AgentEvent::StepStarted { step }` is emitted before each model call so the status bar can
show `n/max` (the loop already knows the count; nothing else could derive it reliably, because
one reply can contain several tool calls).

### D3. Approval is a loop hook; policy lives in the front end
```rust
pub trait Approver: Send + Sync {
    fn decide<'a>(&'a self, call: &'a ApprovalRequest<'a>) -> BoxFuture<'a, Decision>;
}
pub struct ApprovalRequest<'a> { id: &'a ToolCallId, name: &'a ToolName, arguments: &'a ToolArguments, needs_confirmation: bool }
pub enum Decision { Allow, Deny { reason: Option<String> } }   // reason is free text
pub struct AllowAll;                                            // the default
```
`Agent::run_turn_with(message, on_event, &dyn Approver)`; the existing `run_turn`
calls it with `AllowAll` so the REPL, `--once` and tests are untouched. In
`run_tool` the order is: look the tool up → ask the approver (only for tools that
exist; an unknown tool is reported to the model without bothering the user) → emit
`ToolStarted` → run → `ToolFinished`. A denial emits the new
`AgentEvent::ToolDenied { id, name, reason }` and returns the model-facing text
`Error: the user denied this call` (plus ` because: <reason>`). Calls in one reply
are decided one at a time, in order.

The TUI's `TuiApprover` implements the policy: mode `ask` prompts for every call
unless the tool is in the session allow-set; mode `auto` prompts only when
`needs_confirmation`. A prompt sends `Msg::Approval { request, respond: oneshot::Sender<Decision> }`
to the UI and awaits the receiver. The allow-set is cleared by `/reset`. "Allow for
this session" is not offered for `needs_confirmation` tools (they are asked every
time). If the turn is dropped the oneshot is dropped and the UI discards the dialog.
*Alternative:* approval flags on `Tool` only — rejected: the user asked for approving
arbitrary calls, and policy (session allow-set) is UI state. *Alternative:* approver
as a field of `AgentConfig` — rejected: it is not data and would make the config
non-`Clone`/`Debug`.

`Tool::needs_confirmation(&self) -> bool` defaults to `false` (D3 of `agent-tools`).

### D4. Transcript model and rendering cache
`Vec<Entry>` with `Entry::{User(String), Assistant { text, streaming }, Tool(ToolBlock), Notice { level, text }}`.
`ToolBlock { id, name, arguments, state: Waiting|Running|Ok|Failed|Denied, elapsed, output, expanded }`.
Each entry caches its rendered lines keyed by `(revision, width, expanded)`; only the
streaming entry is re-rendered per delta. The view keeps `scroll_from_bottom: u16`
(0 = follow). New lines while `scroll_from_bottom > 0` grow the offset by the added
height so the reader's position is stable and show "↓ N new lines". Only the visible
window of lines is handed to ratatui (no giant `Paragraph`).
Tool blocks are navigable with Tab/Shift-Tab (focus) and Space/Enter-on-focus toggles
expansion; focus is a transcript concept, input keeps the text cursor.

### D5. Markdown: `pulldown-cmark` events → styled lines, own wrapping
A small renderer maps `pulldown-cmark` events to `ratatui::text::Line`s: headings
(bold + colour, `#` markers dropped), emphasis, strong, inline code, fenced/indented
code (dim background, **hard-wrapped by columns, never word-wrapped, never merged
with neighbours**), bullet/ordered lists with hanging indents, block quotes (`│`
gutter), rules, links (text then `(url)` in dim), everything else as plain text. Tables,
images and HTML are rendered as their source text (readable, not pretty).
Word wrapping is our own greedy wrap over styled spans using `unicode-width` because
ratatui's `Paragraph` cannot tell us the wrapped height without an unstable feature,
and the scroll model needs it. Streaming: the text is re-parsed in full on each delta;
`pulldown-cmark` is forgiving of an unclosed fence (it treats the rest as code), which
gives the "partial code is code" behaviour for free, and a property test checks that
rendering any prefix never panics and that rendering the final text equals rendering it
in one go. *Alternative:* `tui-markdown` — rejected: ties us to its ratatui version and
styling, and cannot wrap code the way we want.

### D6. Sanitising all model/tool text
All text that originates from the model, a tool or the server (assistant text, tool
arguments and results, error messages, the reason echoed back) is passed through
`text::sanitise` before layout: `\n` is kept, `\t` becomes four spaces, `\r` is dropped,
other C0/C1 controls and DEL become the visible Control Pictures glyph (U+2400 + code, U+2421 for DEL).
That makes escape-sequence injection impossible by construction, since no layout code
ever sees a raw control character.

### D7. Input widget written in-tree
`InputBuffer` holds `Vec<String>` lines and a (line, column-in-chars) cursor; supports
insert, backspace/delete, arrows, Home/End, Ctrl-A/E/K/U/W, Alt-Enter/Shift-Enter/Ctrl-J
for newline, Up/Down history (first/last line only). Rendering uses `unicode-width`;
the input box grows up to 6 lines then scrolls. Bracketed paste (`Event::Paste`) inserts
text verbatim. `PushKeyboardEnhancementFlags(DISAMBIGUATE_ESCAPE_CODES)` is requested
when the terminal supports it so Shift-Enter is distinguishable; Alt-Enter and Ctrl-J
work everywhere. *Alternative:* `tui-textarea` — rejected: its latest release targets an
older ratatui; an in-tree buffer is ~200 lines and fully testable.

### D8. Terminal guard
`TerminalGuard::enter()` enables raw mode, alternate screen, mouse capture, bracketed
paste; `Drop` undoes each of them (ignoring errors). A panic hook is installed that
restores the terminal first and then calls the previous hook, so the message is readable.
Mouse capture is needed for wheel scrolling; the cost is that text selection needs
Shift-drag in most terminals (documented in `/help` and the README). `terminal.rs` is the
only file that touches the real terminal and is excluded from coverage like other
hardware-bound files.

### D9. Raw mode means Ctrl-C is a key, not a signal
In the TUI Ctrl-C arrives as a key event (`KeyCode::Char('c')` + CONTROL) so the REPL's
`Notify`/SIGINT plumbing is not used. Behaviour: turn running → `Effect::Cancel` (the
runtime drops the turn future; `App` marks the turn cancelled, closes any open approval
dialog and un-finishes running tool blocks as "cancelled"); idle → notice with the hint.

### D10. Front-end selection and CLI
`cli::Args` gains `--plain` (bool) and `--approval <ask|auto>` (`clap::ValueEnum`,
default `ask`). `main.rs` calls a pure `choose_front_end(once, plain, stdin_is_tty,
stdout_is_tty, tui_compiled) -> FrontEnd` (`std::io::IsTerminal`), which is unit tested.
The plain path is the existing code, unchanged. The startup reachability check moves
into the front end: the REPL still prints a warning to stderr, the TUI shows a transcript notice
(and must not write to stderr while the screen is active).

### D11. Shared slash commands
`Command::parse(&str) -> Result<Command, UnknownCommand>` and the help text move to a
small `commands` module used by both front ends so the lists cannot drift. The REPL's
output text stays byte-identical (its tests pin it).

### D12. Typed values
No new free-form strings are introduced as settings: `Approval`/`FrontEnd` are enums,
tool names/ids/arguments use `ToolName`/`ToolCallId`/`ToolArguments`, the base URL shown
in the status bar is a `BaseUrl` (already credential-free). The only free text is what
`String` is allowed for: message text, tool output, the denial reason, notices.

### D13. Testing
- Unit: sanitise, wrap (CJK/emoji widths, long words), markdown cases, streaming-prefix
  property, input buffer, `App::update` (scroll-follow, approval flow, cancel), `choose_front_end`.
- Render: `ratatui::backend::TestBackend` buffers converted to text and compared with
  `insta` snapshots for: empty, streaming Markdown, tool block collapsed/expanded,
  approval dialog, narrow (40 col) and wide layouts, error notice.
- Scripted session: `ScriptedBackend` + synthetic `Msg`s through the same runtime
  function used in production, with the terminal abstracted behind a trait so the
  `run` loop is testable (`EventSource` of `Msg`, `TestBackend` terminal).
- Agent loop: tests for `Approver` (deny with/without reason, ordering, several calls,
  `AllowAll`, unknown tool does not ask, drop while pending).
- Binary: existing piped tests remain; add `--plain` accepted, `--approval bogus` is
  exit code 2, piped stdin never emits escape sequences.
- Live: run in a real terminal against Ollama (tmux `send-keys`/`capture-pane` can drive and observe it).

## Risks / Trade-offs

- [Terminals differ on Shift-Enter and bracketed paste] → Alt-Enter and Ctrl-J always work; paste falls back to key events, where a multi-line paste would send early; documented.
- [Re-parsing Markdown on each streamed delta is O(n²) over a long reply] → only the streaming entry is re-parsed; replies are bounded by `max_tokens`; measure in the live task and add a throttle (render at most every ~16 ms) if needed.
- [Mouse capture blocks native text selection] → documented Shift-drag; revisit a `--no-mouse` flag only if it bothers anyone (non-goal now).
- [ratatui minor releases change APIs] → pin `0.30`, keep ratatui calls inside `view.rs`/`terminal.rs`.
- [Single task: a slow `render` blocks key handling] → rendering is cached and bounded to the visible window.
- [A panic inside the turn future leaves the terminal in raw mode] → the guard's `Drop` plus the panic hook cover both unwinding and the hook path.
- [Two crossterm versions in the tree] → use `ratatui::crossterm` re-exports where possible and check `cargo tree -d` in task 1.

## Migration Plan

Additive. The default behaviour changes only for an interactive terminal without
`--once`: it now opens the TUI; `--plain` restores the old REPL. Rollback is `--plain`,
or building without the `tui` feature.

## Open Questions

- Whether to persist input history across sessions (deferred; no spec depends on it).
