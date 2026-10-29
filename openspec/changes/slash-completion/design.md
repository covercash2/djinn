# Design

See `proposal.md` for why/what and `specs/` for the requirements. All changes are in `djinn-agent`.

## Context

- `commands.rs` has `Command::parse` (a `match` on names) and the `HELP` text as a
  `const &str` that the REPL tests pin byte-for-byte.
- The TUI input is an `InputBuffer` with `text()`, cursor position and `layout()`;
  `App::on_key` maps keys to input edits and `submit()`.
- `view::render` lays out transcript, input box and status bar, and draws the approval
  dialog as an overlay.

## Decisions

### D1. A command table next to the enum
```rust
pub struct CommandInfo { pub name: &'static str, pub command: Command, pub description: &'static str }
pub const COMMANDS: &[CommandInfo]   // exit, quit, reset, tools, help — one row per name, in /help order
```
`Command::parse` looks names up in the table (so parsing and completion cannot
disagree). `HELP` stays a literal because the REPL output must not change; a test asserts
that every table entry's name and description appear in `HELP`, in table order. Completion rows
are one per *name*, so `/exit` and `/quit` are separate rows (as in `/help`), both carrying the
description of the command they run.
*Alternative:* generate `HELP` from the table at run time — rejected for now: it would
change the REPL's byte-exact output tests for no user benefit; the sync test gives the same safety.

### D2. Completion state is derived, plus two small pieces of memory
The candidates are a pure function: `completions(text, cursor) -> Option<(prefix, Vec<&CommandInfo-name rows>)>` which returns
`None` unless the text is one line starting with `/`, the cursor is at or before the end
of the first word, and at least one name starts with the typed word (ASCII case-insensitive).
`App` keeps only `selected: usize` and `dismissed: Option<String>` (the input text when Esc was pressed; the
popup stays closed while the text is unchanged, so moving the cursor does not reopen it). `selected` resets to 0
whenever the typed word changes (compared with the previous prefix kept in `App`), and the selection is clamped to the list length when read. Because the list
is recomputed from the input on every use, there is no stale state to invalidate.

### D3. Key routing
In `App::on_key`, after the approval-dialog and tool-focus handling and before the normal
editing keys: when `completions()` is `Some` and the agent is idle and not dismissed,
- `Up`/`Down` change `selected` (wrapping), `Tab` accepts, `Esc` sets `dismissed`,
- `Enter` accepts when the typed word is not exactly the selected name, otherwise falls
  through to `submit()`.
All other keys fall through unchanged. Accepting replaces the whole input with `/<name>`
(the commands take no arguments, so no trailing space) and puts the cursor at the end.
Tab with no popup keeps its current meaning (tool-block selection).

### D4. Drawing
`view::render` draws the popup after the input box, as a bordered block whose bottom edge sits on the
top edge of the input box, left-aligned with it, at most 6 rows, as wide as the longest row plus
padding (clamped to the screen). Rows read `/name  description`; the selected row is reversed.
If the transcript area is too short to hold it the popup is clipped to the rows available
(zero rows → not drawn). The cursor stays in the input box.

### D5. No new typed values
Candidates are `&'static str` names from the table, resolved to `Command` by the existing
parser; nothing user-typed is stored other than in the existing input buffer.

## Risks / Trade-offs

- [Up/Down are overloaded with history recall] → only while the popup is open, which needs
  a leading `/` on a one-line input; Esc closes it and `Up` then recalls as before.
- [Enter completing instead of running may surprise people who type a full prefix like `/h`]
  → `/h` is a prefix of `/help` only, Enter completes to `/help`, a second Enter runs it; the
  popup makes the state visible. Exact names (`/help`) run immediately.
- [Popup hides part of the transcript] → it is only up while a command is being typed.

## Migration Plan

Additive; the REPL is untouched. Rollback is reverting the commit.

## Open Questions

None that change the specs or tasks.
