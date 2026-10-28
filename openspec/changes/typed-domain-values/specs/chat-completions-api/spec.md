## ADDED Requirements

### Requirement: Request values are validated when parsed
A chat request SHALL be rejected with status 400 and a JSON error that names the
offending field and the rule it broke when it contains: a tool name that is not a
valid tool name, a tool-call id (in an assistant tool call or a `tool` message)
that is empty or has whitespace, tool-call arguments that are not a JSON object,
an empty stop string, a `temperature` outside 0 to 2, a `top_p` not above 0 and
at most 1, or a `max_tokens` of 0. See `domain-values` for the exact rules.

#### Scenario: Invalid tool name
- **WHEN** a request declares a tool named `read file`
- **THEN** the response is 400 and the error message names the tool name and the allowed characters

#### Scenario: Empty stop string
- **WHEN** a request sets `stop` to `["END", ""]`
- **THEN** the response is 400 and the error message says a stop string cannot be empty

#### Scenario: Temperature out of range
- **WHEN** a request sets `temperature` to `-1` or `5`
- **THEN** the response is 400 and the error message names `temperature`

#### Scenario: Zero token limit
- **WHEN** a request sets `max_tokens` to `0`
- **THEN** the response is 400 and the error message names `max_tokens`

#### Scenario: Replayed call arguments must be an object
- **WHEN** an assistant message in the history carries a tool call whose `function.arguments` is `[1,2]`
- **THEN** the response is 400

#### Scenario: Empty arguments mean no arguments
- **WHEN** an assistant message in the history carries a tool call whose `function.arguments` is the empty string
- **THEN** the request is accepted and the call is treated as having an empty arguments object

#### Scenario: Valid request is unaffected
- **WHEN** a request uses only valid values, including `stop` as a single string and `temperature` `0`
- **THEN** it behaves exactly as before this requirement
