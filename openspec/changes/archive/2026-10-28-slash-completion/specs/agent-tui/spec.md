# agent-tui Specification

## ADDED Requirements

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

