# agent-tools Specification

## Purpose
Defines the contract for agent tools and the three built-in tools, all read-only
and safe to run without confirmation, so the agent can do useful things on day one
without risk of changing the machine.

## Requirements

### Requirement: Tool contract
Every tool SHALL have a unique, valid tool name (see `domain-values`), a
description for the model, and a JSON-schema description of its parameters, and
SHALL run asynchronously taking JSON arguments and returning text or an error
message. The agent SHALL advertise exactly the registered tools to the model on
every request.

#### Scenario: Tools are advertised
- **WHEN** three tools are registered
- **THEN** every model request lists those three with their names, descriptions, and parameter schemas

#### Scenario: Duplicate names rejected
- **WHEN** two tools with the same name are registered
- **THEN** registration fails with an error naming the duplicate

#### Scenario: Invalid name cannot be used
- **WHEN** a tool is defined with the name `read file`
- **THEN** the name is rejected when it is created, so the tool can never be registered or advertised

#### Scenario: Built-in names are valid
- **WHEN** the built-in tools `current_time`, `calculator`, and `read_file` are registered
- **THEN** registration succeeds

### Requirement: Built-in tools are read-only
The built-in tools SHALL NOT write or delete files, run processes, or use the
network. Adding a tool that does any of these SHALL require a separate change that
also specifies user confirmation.

#### Scenario: No side effects
- **WHEN** each built-in tool is run with valid arguments
- **THEN** the file system and network state are unchanged afterwards

### Requirement: current_time tool
`current_time` SHALL take no required arguments and SHALL return the current date
and time in UTC as an RFC 3339 string.

#### Scenario: Returns a parseable time
- **WHEN** the tool runs
- **THEN** the result parses as an RFC 3339 timestamp within a few seconds of the system clock

### Requirement: calculator tool
`calculator` SHALL take an `expression` string and return its numeric value. It
SHALL support decimal numbers, `+`, `-`, `*`, `/`, parentheses, and unary minus
with the usual precedence, and SHALL reject anything else with an error message
instead of evaluating it.

#### Scenario: Precedence and parentheses
- **WHEN** the expression is `2 + 3 * (4 - 1)`
- **THEN** the result is `11`

#### Scenario: Division by zero
- **WHEN** the expression is `1 / 0`
- **THEN** the tool returns an error saying division by zero

#### Scenario: Not arithmetic
- **WHEN** the expression is `system("ls")` or `2 ** 3`
- **THEN** the tool returns an error saying the expression is not supported and evaluates nothing

#### Scenario: Missing argument
- **WHEN** `expression` is absent
- **THEN** the tool returns an error naming the missing parameter

### Requirement: read_file tool
`read_file` SHALL take a `path` and return the file's text, restricted to files
inside a configured root directory (default: the current directory). It SHALL
resolve symbolic links and `..` before checking, SHALL refuse paths that resolve
outside the root, SHALL refuse non-UTF-8 files and directories, and SHALL return at
most 64 KiB, saying so when it truncates.

#### Scenario: Reads a file under the root
- **WHEN** `path` is `README.md` and it exists under the root
- **THEN** the tool returns its text

#### Scenario: Escape with dot-dot
- **WHEN** `path` is `../secrets.txt`
- **THEN** the tool returns an error saying the path is outside the allowed directory and reads nothing

#### Scenario: Escape through a symlink
- **WHEN** a symlink under the root points at a file outside it
- **THEN** reading the symlink returns the outside-the-root error

#### Scenario: Absolute path outside the root
- **WHEN** `path` is `/etc/passwd`
- **THEN** the tool returns the outside-the-root error

#### Scenario: Large file
- **WHEN** the file is larger than 64 KiB
- **THEN** the result contains the first 64 KiB and a note stating the file was truncated and its total size

#### Scenario: Binary file
- **WHEN** the file is not valid UTF-8
- **THEN** the tool returns an error saying it is not a text file

#### Scenario: Missing file
- **WHEN** the file does not exist
- **THEN** the tool returns a not-found error that does not reveal paths outside the root
