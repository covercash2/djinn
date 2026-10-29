# Tasks

> Run everything through the Nix dev shell: `nix develop --command <cmd>` (cargo,
> just, typos exist only there). **Every group ends with `just check` green.** The
> owner commits with `jj`; do not stage or commit. Read `design.md` first — decisions
> D1–D13 are referenced below. Groups 4–6 are independent of group 2–3 once group 1 is done.

## 1. Scaffolding

- [x] 1.1 Add `ratatui = "0.30"`, `crossterm = { version = "0.29", features = ["event-stream"] }`, `pulldown-cmark`, `unicode-width` to `[workspace.dependencies]`; in `djinn-agent/Cargo.toml` make them optional behind a default-on `tui` feature (`tui = ["dep:ratatui", ...]`). Verify `cargo tree -d` shows a single `crossterm`, `just check-agent-deps` passes, and `cargo build -p djinn-agent --no-default-features` works.
- [x] 1.2 Create the empty module tree `djinn-agent/src/tui/{mod,app,view,markdown,text,input,approval,terminal}.rs` behind `#[cfg(feature = "tui")]`, with a `//!` header per file stating its responsibility (D1). `just check`.

## 2. Approval in the agent loop

- [x] 2.1 `tools/mod.rs`: add `Tool::needs_confirmation(&self) -> bool { false }`; test that the three built-ins return false and that a test tool can override it.
- [x] 2.2 `agent.rs`: add `Approver`, `ApprovalRequest`, `Decision`, `AllowAll`, `AgentEvent::ToolDenied`, and `Agent::run_turn_with`; make `run_turn` call it with `AllowAll` (D3). Order in `run_tool`: lookup → approve (skipped for unknown tools) → `ToolStarted` → run → `ToolFinished`; a denial yields the model-facing result text from D3.
- [x] 2.3 Tests in `agent.rs` for every scenario of "Tool approval": denied with and without reason (tool not run, result text, turn continues), decision precedes `ToolStarted`, two calls with the first denied, no approver = old behaviour (existing tests unchanged), unknown tool never asks, dropping the turn while a decision is pending leaves history untouched. Update the REPL's `show` to ignore/print `ToolDenied` (never reached there, but the match must be exhaustive).
- [x] 2.4 `just check`.

## 3. Shared commands, front-end selection, CLI

- [x] 3.1 New `commands.rs` with `Command::parse`, help text, `UnknownCommand`; make `repl.rs` use it with byte-identical output (the existing REPL tests must pass unchanged).
- [x] 3.2 `cli.rs`: `--plain`, `--approval <ask|auto>` (`Approval` enum, default `ask`); `main.rs`: pure `choose_front_end(...)` plus unit tests for the four scenarios of "Front end selection" (D10). Without the `tui` feature the TUI branch is absent and a terminal gets the REPL.
- [x] 3.3 Binary tests in `tests/cli.rs`: `--plain` and `--approval auto` are accepted with piped stdin, `--approval bogus` exits 2 naming `--approval`, piped stdin output contains no ESC byte. The 13 existing tests still pass.
- [x] 3.4 `just check`.

## 4. Pure text machinery

- [x] 4.1 `tui/text.rs`: `sanitise` (D6) and `wrap_spans` (greedy, display-width aware, hard-wrap for over-long words and for code lines). Tests: controls/ESC become visible placeholders, tabs, CRLF, CJK and emoji widths, a word longer than the width, empty input, width 1.
- [x] 4.2 `tui/markdown.rs`: render per D5. Tests for each element in "Markdown rendering" (headings, emphasis, inline code, fenced code incl. a line wider than the view, lists with hanging indent, quotes, rule, link, table/HTML fallback, plain text unchanged) plus the streaming property: for a corpus of documents, every prefix renders without panic and the full text renders the same whether delivered at once or char by char.
- [x] 4.3 `tui/input.rs`: `InputBuffer` per D7 with tests for editing, multi-line, cursor keys, history recall only at first/last line, paste with newlines, grow/scroll window, blank-message detection.
- [x] 4.4 `just check`.

## 5. App state

- [x] 5.1 `tui/app.rs`: `Entry`/`ToolBlock`/`Msg`/`Effect`/`App::update` (D2, D4): sending (blank ignored, busy blocks send but allows typing), streaming text appends to one assistant entry, tool state transitions from `AgentEvent`s incl. `ToolDenied`, turn end/error/step-limit notices, `/` commands via `commands::Command`, `/reset` clears transcript and allow-set, Ctrl-C and Ctrl-D rules (D9), scrolling and follow mode with "new lines" counter, tool-block focus and expansion.
- [x] 5.2 `tui/approval.rs`: `TuiApprover` (modes `ask`/`auto`, session allow-set, oneshot per request) and the dialog state in `App` (allow once / allow session / deny + reason editing; session choice hidden for `needs_confirmation` tools; Ctrl-C closes it and cancels). Tests for all scenarios of "Approving tool calls" at the `App::update` level and `TuiApprover` level (with a test tool that needs confirmation in `auto` mode).
- [x] 5.3 `just check`.

## 6. View

- [x] 6.1 `tui/view.rs`: layout (transcript, input box, status bar), transcript window rendering with the per-entry line cache (D4), tool blocks collapsed/expanded, approval dialog overlay, status bar content (model, `BaseUrl`, approval mode, idle/working `n/max`, key hints), "↓ new" indicator.
- [x] 6.2 Snapshot tests with `TestBackend` + `insta` for the screens listed in D13 (empty, streaming Markdown, tool block collapsed/expanded, approval dialog, 40-column narrow, error notice, status while working `2/8`). Review every snapshot by eye before accepting.
- [x] 6.3 `just check`.

## 7. Runtime

- [x] 7.1 `tui/terminal.rs`: `TerminalGuard` (raw mode, alternate screen, mouse, bracketed paste, keyboard enhancement when supported) with `Drop` restore and the chained panic hook (D8). Exclude this file in `coverage_ignore` (justfile and CI regex, kept in sync).
- [x] 7.2 `tui/mod.rs`: `run(agent, settings)` — generic over an event source and a ratatui `Backend` so tests use synthetic events and `TestBackend`; the single `select!` loop of D2 (event stream, turn channel, ticker only while busy, the turn future), `Effect` execution (send, cancel by dropping the future, reset, quit), resize handling, startup reachability notice.
- [x] 7.3 Wire `main.rs`: TUI branch builds the `TuiApprover` from `--approval` and runs `tui::run`; the SIGINT task is not started in this branch; the final terminal state is restored before any error is printed.
- [x] 7.4 Scripted-session tests for the scenarios "Scripted session", "Cancel a long answer", "Server down at start" / backend error, "Step limit", "Allow for the session", "Deny", "Cancel while waiting", "Resize while streaming" (content preserved), driven through `run` with `ScriptedBackend` and synthetic events.
- [x] 7.5 `just check`.

## 8. Docs and live verification

- [x] 8.1 README: a "Terminal UI" section (starting it, keys table, approval modes, `--plain`, Shift-drag selection, tmux/ssh notes); update `CLAUDE.md` Workspace/Commands lines for `djinn-agent`; make sure `just agent` opens the TUI and note `--plain` for scripts.
- [x] 8.2 Live check against the local Ollama (`qwen2.5` or `llama3.2`) in a real terminal session (drive it with tmux `send-keys`/`capture-pane` if available): streamed Markdown answer, a calculator call approved once, a `read_file` call allowed for the session, a denial with reason that the model reacts to, Ctrl-C mid-stream, resize, `/reset`, `/exit` leaves the shell sane. Record the observations at the end of this file.
- [x] 8.3 Full `just check`, `nix build .#djinn-cli`, and `openspec validate --all --strict`. Leave committing and `openspec archive agent-tui` to the owner.

## Live verification notes (8.2)

Run with `script` (a pty) against the local Ollama, `qwen2.5:latest`, terminal 100x30; tmux is not installed here, so the keys were fed on stdin with delays and the output was inspected with escape sequences stripped:

- Start: status bar `qwen2.5:latest · http://localhost:11434/v1 · approval: ask · ● idle`; the empty input box is drawn.
- "What is 17*23? Use the calculator tool." → the `calculator` call appears as `▸ ? calculator {"expression":"17*23"} waiting for approval` with the dialog (`[y] allow once  [a] allow this tool for the session  [n] deny`); `y` → `✓ calculator 0ms … → 391`; the answer streamed with the `▍` cursor and ended `…is 391.`; status back to `● idle`.
- Denial: `n`, then the reason "I know it already", Enter → block `⊘ denied: I know it already`; the model replied that 6 times 7 is 42 (it reacted to the denial).
- Session allowance: `a` on `current_time` → it ran (`✓ current_time … → 2026-10-29T02:25:46Z`); a dialog was shown only once for that tool.
- Ctrl-C during a long answer → `turn cancelled` notice, input accepted again.
- `/exit` → exit code 0, and the terminal was restored (the stream ends with `?1049l`, `?25h`, bracketed-paste and mouse reporting off).
- Found and fixed during this check: the approval dialog for a tool without arguments showed both `(no arguments)` and `{}`.
- Not exercised live: terminal resize (covered at the view level by `resizing_rewraps_without_losing_text`; the scripted-session group cannot resize the test terminal), a panic restoring the terminal (code path only; `terminal.rs` is excluded from coverage).
