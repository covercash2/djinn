# Tasks

> Run everything through the Nix dev shell: `nix develop --command <cmd>`. **Every
> group ends with `just check` green.** The owner commits with `jj`; do not stage or
> commit. Read `design.md` first (D1–D5).

## 1. Command table

- [x] 1.1 `commands.rs`: add `CommandInfo` and `COMMANDS` (D1); make `Command::parse` use it; keep `HELP` byte-identical. Tests: every table entry parses (name and aliases), names are unique, and `HELP` contains every name, alias and description (so the two cannot drift). Existing REPL tests pass unchanged.
- [x] 1.2 `just check`.

## 2. Completion logic and key handling

- [x] 2.1 `tui/completion.rs` (new): pure `completions(text, cursor)` per D2 with tests for every scenario of "Slash command completion" (`/` lists all in `/help` order, `/re`, `/qu` alias, case-insensitive `/RE`, `/zzz` none, second line, cursor after the first space, empty input, plain text, leading space before the slash → none).
- [x] 2.2 `tui/app.rs`: `selected`/`dismissed`/previous-prefix state and key routing per D3. Tests for every scenario of "Choosing a completion": Tab accepts, Enter completes then runs, exact name runs at once, wrapping navigation, Esc dismisses and a further keystroke re-opens it, Up recalls history when no popup, Tab selects tool blocks when no popup, no popup while working, selection resets to the first row when the prefix changes.
- [x] 2.3 `just check`.

## 3. Drawing

- [x] 3.1 `tui/view.rs`: draw the popup per D4. `insta` snapshots: all commands, narrowed to one, selected row moved, narrow (40 col) terminal, a very short terminal (clipped, no panic). Review each snapshot by eye.
- [x] 3.2 Scripted-session test through `tui::run` (existing harness): type `/to`, Enter, Enter → the tools listing appears in the transcript and no model request was made.
- [x] 3.3 `just check`.

## 4. Docs and live check

- [x] 4.1 README keys table (popup, Tab, Esc) and the `/help` key list in `app::help_text`; update the `help_text` test if it pins the text.
- [x] 4.2 Live check in a pty as in the `agent-tui` notes (type `/`, `/re`, Tab, Enter, Esc) and record the observations at the end of this file; `openspec validate --all --strict`. Leave committing and archiving to the owner.

## Live verification notes (4.2)

Run in a pty with `script` (80x24), keys fed on stdin with delays, output inspected with escape sequences stripped:

- `/` → the `commands` popup with all five rows appeared above the input box; `/re` narrowed it to `/reset  forget the conversation`.
- `Tab` filled in `/reset`; `Esc` closed the popup; `Enter` then ran `/reset` (a name typed in full runs at once).
- `/to`, `Enter` → input became `/tools` without running it; second `Enter` ran it and the tool listing (`calculator: …`) appeared in the transcript.
- `/q`, `Down` (a single row wraps onto itself), `Enter` → `/quit`, `Enter` → exit code 0 with the terminal restored.
- A cursor-position change after Esc must not reopen the popup: this was found by a unit test, not live, and the dismissal is therefore tied to the input text (design D2 updated).
