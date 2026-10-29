# agent-tools Specification

## ADDED Requirements

### Requirement: Tools declare whether they need confirmation
A tool SHALL be able to declare that its calls need the user's confirmation; by
default a tool does not. Front ends that support approval SHALL always ask for tools
that declare it, in every approval mode, and SHALL be able to ask for all other tools
as well. The three built-in tools do not declare it.

#### Scenario: Default
- **WHEN** a tool is defined without stating anything about confirmation
- **THEN** it does not need confirmation

#### Scenario: A tool that needs confirmation in automatic mode
- **WHEN** the TUI runs with `--approval auto` and the model calls a tool that declares it needs confirmation
- **THEN** the user is asked before it runs, while calls of tools that do not declare it run without asking
