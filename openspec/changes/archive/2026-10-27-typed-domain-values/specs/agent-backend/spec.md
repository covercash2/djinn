## ADDED Requirements

### Requirement: Connection settings are validated at start-up
The agent SHALL validate its base URL, model name, and API key when its
arguments and environment are read (see `domain-values`). An invalid one SHALL
make the agent exit non-zero with a message that names the setting and the rule
before any request is made. The API key SHALL NOT appear in debug output, logs,
or error messages.

#### Scenario: Bad base URL
- **WHEN** the agent is started with `--base-url localhost:8080`
- **THEN** it exits non-zero, names `--base-url`, and sends no request

#### Scenario: Empty model name
- **WHEN** the agent is started with `--model ""`
- **THEN** it exits non-zero and names `--model`

#### Scenario: Key never printed
- **WHEN** the agent is started with an API key and a request fails
- **THEN** neither the error output nor any log line contains the key text

### Requirement: Malformed identifiers from servers are reported
If a server's response contains a tool call whose name or id is not valid, or a
tool call whose arguments are not a JSON object, the backend SHALL report a
malformed-response failure that names the problem, and SHALL NOT pass the call
on. Text already streamed remains visible.

#### Scenario: Server sends a bad tool name
- **WHEN** a server returns a tool call named `my tool`
- **THEN** the backend reports a malformed response naming the tool name, and no tool is run

#### Scenario: Server omits ids
- **WHEN** a server omits tool-call ids
- **THEN** the backend still generates valid ids and the call is accepted

#### Scenario: Unusual but valid response
- **WHEN** a server omits the `object` field of a streamed chunk
- **THEN** the chunk is accepted and the reply is assembled normally
