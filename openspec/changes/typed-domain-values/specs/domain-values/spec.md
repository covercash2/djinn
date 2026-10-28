# domain-values Specification

## Purpose
Defines the kinds of values djinn treats as more than free text (identifiers,
paths, addresses, secrets, tuning parameters), what makes each one valid, and
the guarantee that an invalid one is rejected where it enters the program. Every
other capability relies on this so it does not need to re-check the same rules.

## ADDED Requirements

### Requirement: Values are validated where they enter the program
Every domain value SHALL be checked when it is created in code and when it is
read from outside the program (an HTTP request body, a TOML config, a
command-line argument or environment variable, or a response from another
server). An invalid value SHALL be rejected there with an error that names the
kind of value and the rule it broke, and SHALL NOT reach later code.

#### Scenario: Rejected at the HTTP boundary
- **WHEN** a request body contains a value that breaks its rule
- **THEN** the request fails before any handler runs, and the error names the offending value kind and rule

#### Scenario: Rejected at config load
- **WHEN** a TOML config contains a value that breaks its rule
- **THEN** loading the config fails with an error that names the key and the rule

#### Scenario: Rejected at argument parsing
- **WHEN** a command-line argument breaks its rule
- **THEN** the program exits non-zero with a usage error naming the argument, before doing any other work

### Requirement: Serialized form is unchanged
A validated value SHALL serialize and deserialize exactly like the plain
string, number, or path it stands for. Introducing or changing a validated
type SHALL NOT change the JSON bodies of the HTTP API or the TOML config file
format.

#### Scenario: Round trip
- **WHEN** a valid value is serialized and read back
- **THEN** the result equals the original, and the serialized form is the bare string or number

#### Scenario: Existing configs still load
- **WHEN** the shipped configs under `configs/` are loaded
- **THEN** all of them load without edits

### Requirement: Tool identifiers
A tool name SHALL be 1 to 64 characters from ASCII letters, digits, `_` and
`-`. A tool-call id SHALL be 1 to 256 characters with no whitespace or control
characters. Tool-call arguments SHALL be text that parses as a JSON object; text
that is empty or only whitespace SHALL be accepted and mean an empty object.

#### Scenario: Valid and invalid tool names
- **WHEN** the names `calculator`, `read_file`, and `get-time` and the names `read file`, `` (empty), `a.b`, and a 65-character name are checked
- **THEN** the first three are accepted and the other four are rejected

#### Scenario: Arguments must be a JSON object
- **WHEN** tool arguments are `{"expression":"2+2"}`, `{}`, ``, `[1]`, `"x"`, or `{"a":`
- **THEN** the first three are accepted and the others are rejected

#### Scenario: Arguments keep their exact text
- **WHEN** valid arguments with unusual spacing are stored and serialized again
- **THEN** the text is byte-for-byte what was given

### Requirement: Model and config identifiers
A model identifier SHALL be 1 to 256 characters with no control characters and
no leading or trailing whitespace. A config name SHALL be a relative
`/`-separated path of segments from ASCII letters, digits, `.`, `_` and `-`,
with no `.` or `..` segment and no `.toml` suffix.

#### Scenario: Valid identifiers
- **WHEN** `llama3.2:latest` (model id) and `model/qwen2_5_instruct` (config name) are checked
- **THEN** both are accepted

#### Scenario: Config name escapes the config directory
- **WHEN** the config name `../server/test` is checked
- **THEN** it is rejected

#### Scenario: Config name with extension
- **WHEN** the config name `test.toml` is checked
- **THEN** it is rejected with a message saying to give the name without `.toml`

### Requirement: Hub identifiers
A Hub repository id SHALL have the form `owner/name`, each part 1 to 96
characters from ASCII letters, digits, `.`, `_` and `-`, and neither part `.` or
`..`. A Hub revision SHALL be 1 to 128 characters with no whitespace or control
characters. A weights file SHALL be a relative path with `/` separators, 1 to 256
characters, and no empty, `.` or `..` segment.

#### Scenario: Valid Hub identifiers
- **WHEN** `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (repository), `main` (revision), and `qwen2.5-1.5b-instruct-q4_k_m.gguf` (weights file) are checked
- **THEN** all are accepted

#### Scenario: Path escapes are rejected
- **WHEN** the weights files `../secret.gguf` and `/etc/passwd` are checked
- **THEN** both are rejected

#### Scenario: Malformed repository id
- **WHEN** the repository ids `no-slash`, `a/b/c`, and `owner/` are checked
- **THEN** each is rejected

### Requirement: Text roles
A stop sequence SHALL be 1 to 256 bytes long, and a list of stop sequences
SHALL hold at most 4. A system prompt SHALL contain at least one non-whitespace
character. A completion prompt SHALL be any text, including the empty string.

#### Scenario: Empty stop sequence
- **WHEN** the stop sequence `` (empty) is checked
- **THEN** it is rejected, because it would match before any text is produced

#### Scenario: Too many stop sequences
- **WHEN** a list of 5 stop sequences is checked
- **THEN** it is rejected, and a list of 4 is accepted

#### Scenario: Blank system prompt
- **WHEN** a system prompt of only spaces and newlines is checked
- **THEN** it is rejected

#### Scenario: Empty completion prompt
- **WHEN** the completion prompt `` (empty) is checked
- **THEN** it is accepted

### Requirement: Sampling parameters
A temperature SHALL be a finite number from 0 to 2 inclusive. A nucleus
sampling probability (`top_p`) SHALL be a finite number greater than 0 and at
most 1. A token limit SHALL be a whole number of at least 1. NaN and infinite
values SHALL be rejected. The built-in defaults SHALL satisfy these rules.

#### Scenario: Boundaries
- **WHEN** temperatures 0, 0.7, 2 and -0.1, 2.1, NaN are checked
- **THEN** the first three are accepted and the last three are rejected

#### Scenario: top_p boundaries
- **WHEN** `top_p` values 0.01 and 1 and 0 and 1.5 are checked
- **THEN** the first two are accepted and the last two are rejected

#### Scenario: Zero token limit
- **WHEN** a token limit of 0 is checked
- **THEN** it is rejected

#### Scenario: Defaults are valid
- **WHEN** a configuration is built from defaults only
- **THEN** every sampling parameter in it satisfies its rule

### Requirement: Connection settings of the agent
The agent's service base URL SHALL be an absolute `http` or `https` URL with a
host and with no user information, query, or fragment. An API key SHALL be 1 or
more visible ASCII characters (no whitespace or control characters, so it is always
a valid header value), and SHALL NOT appear
in debug output, logs, or error messages.

#### Scenario: Valid and invalid base URLs
- **WHEN** `http://localhost:8080/v1` and `https://api.example.com/v1`, and `localhost:8080`, `ftp://host/v1`, `http://user:pass@host/v1`, and `http://host/v1?x=1` are checked
- **THEN** the first two are accepted and the others are rejected

#### Scenario: Secret is redacted
- **WHEN** a structure holding an API key is formatted for debugging
- **THEN** the key text does not appear in the output

### Requirement: Constraints are published in schemas
The rules of every validated value that appears in a config file or an HTTP
body SHALL be expressed in the generated JSON Schema files under `configs/` and
in the OpenAPI document as far as the schema language allows (pattern, length,
and numeric bounds). Regenerating the schemas SHALL leave the working tree
unchanged.

#### Scenario: Schema shows the rule
- **WHEN** `configs/model/model.schema.json` is read
- **THEN** the repository id, revision, and weights file properties carry their pattern or length constraints

#### Scenario: OpenAPI shows the rule
- **WHEN** `/api-doc/openapi.json` is fetched
- **THEN** the chat request's `temperature` property has minimum 0 and maximum 2

### Requirement: Free text stays free text
Values that are genuinely arbitrary text SHALL remain plain strings: message
content, tool output, generated text, completion output, error messages, the
ignored `model` field of a chat request, and the values inside a tool's
arguments (each tool validates its own parameters).

#### Scenario: Arbitrary content is accepted
- **WHEN** a chat message has content that is empty, very long, or contains control characters
- **THEN** it is accepted as before

#### Scenario: Request model field is not checked
- **WHEN** a chat request sets `model` to any string, including the empty string
- **THEN** the request is not rejected because of it
