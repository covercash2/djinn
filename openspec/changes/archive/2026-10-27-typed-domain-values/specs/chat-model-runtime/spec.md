## ADDED Requirements

### Requirement: Model source settings are validated at config load
The Hub repository id, revision, weights file, and tokenizer repository of a
model config SHALL be checked when the config is loaded (see `domain-values`).
A config with an invalid value SHALL fail to load with an error that names the
key and the rule, and SHALL NOT start a download.

#### Scenario: Malformed repository id
- **WHEN** a model config sets `repo_id = "not a repo"`
- **THEN** loading fails and the error names `repo_id`

#### Scenario: Path escape in weights file
- **WHEN** a model config sets `weights_file = "../../etc/passwd"`
- **THEN** loading fails and the error names `weights_file`, and nothing is fetched

#### Scenario: Shipped configs remain valid
- **WHEN** every config under `configs/model/` is loaded
- **THEN** all load successfully

### Requirement: Extracted tool calls have valid names
A tool call extracted from generated text SHALL be returned only if its name is
a valid tool name. A call with an invalid name SHALL be treated as invalid
markup: kept as plain text and not returned as a call.

#### Scenario: Invalid name in model output
- **WHEN** the output is `<tool_call>{"name":"do it!","arguments":{}}</tool_call>`
- **THEN** no call is returned and the text is preserved as content

#### Scenario: Valid name unchanged
- **WHEN** the output is a valid `calculator` call
- **THEN** it is returned exactly as before
