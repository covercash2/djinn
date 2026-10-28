# agent-backend Specification

## Purpose
Defines how the agent obtains model output from an inference service, so that
the same agent works against any OpenAI-compatible server (the djinn server,
Ollama, llama.cpp, vLLM) without code changes.

## Requirements

### Requirement: Backend abstraction
The agent SHALL obtain model replies only through a backend abstraction that, given
the conversation, tool declarations, and generation options, returns a stream of
text fragments followed by a final result containing the complete text, any tool
calls, and a finish reason. The agent loop SHALL NOT depend on any particular
transport.

#### Scenario: A scripted backend drives the agent
- **WHEN** the agent is run with an in-memory backend that returns scripted replies
- **THEN** it behaves exactly as with a network backend, with no network access

### Requirement: OpenAI-compatible HTTP backend
The agent SHALL provide a backend that talks to a server's `/chat/completions`
endpoint under a configurable base URL, with a configurable model name and an
optional bearer token, using streaming responses. The default base URL SHALL be
`http://localhost:8080/v1` and the default model name SHALL be `default`.

#### Scenario: Works against the djinn server
- **WHEN** the base URL points at a running djinn server
- **THEN** a question is answered with streamed text

#### Scenario: Works against Ollama
- **WHEN** the base URL is `http://<host>:11434/v1` and the model name is a tool-capable Ollama model
- **THEN** a question is answered and tool calls are received

#### Scenario: Token is sent when configured
- **WHEN** an API key is configured
- **THEN** each request carries it as a bearer `Authorization` header, and when none is configured no such header is sent

### Requirement: Streaming and tool-call assembly
The backend SHALL deliver text fragments as they arrive. It SHALL assemble tool
calls from the stream, including servers that send a call's arguments in several
fragments, and SHALL return each call only once complete with valid-JSON
arguments.

#### Scenario: Fragmented arguments
- **WHEN** a server streams a tool call's argument string in three pieces
- **THEN** the backend returns one call whose arguments equal the concatenation, parsed as JSON

#### Scenario: Server without streaming tool calls
- **WHEN** a server delivers the complete call in one chunk
- **THEN** the backend returns the same call result

#### Scenario: Missing call id
- **WHEN** a server omits tool-call ids
- **THEN** the backend generates unique ids so results can be matched to calls

### Requirement: Failure reporting
Backend failures SHALL be reported to the caller as distinct, readable errors:
connection failure (including the URL tried), HTTP error status (with the status
and the server's error message when present), a stream that ends before completion,
and malformed response data. The agent SHALL NOT retry automatically.

#### Scenario: Server not running
- **WHEN** nothing listens at the base URL
- **THEN** the error says the connection failed and names the URL

#### Scenario: Error status
- **WHEN** the server answers 400 with `{"error":{"message":"context too long"}}`
- **THEN** the reported error includes status 400 and the message

#### Scenario: Stream cut off
- **WHEN** the connection closes before the final chunk
- **THEN** the error says the stream ended early, and partial text already shown remains visible

### Requirement: Independent of the inference runtime
The agent crate SHALL NOT depend, directly or transitively, on candle,
`djinn-models`, or `djinn-server`. It MAY depend on `djinn-core` for the shared
chat wire types.

#### Scenario: Dependency check
- **WHEN** `cargo tree -p djinn-agent` is run
- **THEN** the output contains none of `candle-core`, `djinn-models`, `djinn-server`

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
