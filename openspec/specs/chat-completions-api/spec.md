# chat-completions-api Specification

## Purpose
Gives `djinn-server` a chat interface compatible with the widely used
OpenAI-style chat-completions protocol, including tool calls, so any client that
speaks that protocol (including the djinn agent) can use the loaded model.

## Requirements

### Requirement: Chat completion request
`POST /v1/chat/completions` SHALL accept JSON with `messages` (ordered list; each
has a `role` of `system`, `user`, `assistant`, or `tool`, and text `content`),
optional `tools`, `stream`, `max_tokens`, `temperature`, `top_p`, `stop`, `seed`,
and `model`. Unknown fields SHALL be ignored. A request with an empty `messages`
list or an unknown `role` SHALL be rejected with status 400.

#### Scenario: Minimal request
- **WHEN** a client posts `{"messages":[{"role":"user","content":"hi"}]}` with `Content-Type: application/json`
- **THEN** the response is 200 with a chat completion body

#### Scenario: Empty messages
- **WHEN** a client posts `{"messages":[]}`
- **THEN** the response is 400 with a JSON error naming the problem

#### Scenario: Unknown role
- **WHEN** a client posts a message with role `robot`
- **THEN** the response is 400

#### Scenario: Unknown extra field
- **WHEN** a client includes `"logprobs": true`
- **THEN** the field is ignored and the request succeeds

### Requirement: Non-streaming response shape
When `stream` is absent or false, the response SHALL be a JSON object with
`object` = `chat.completion` and a `choices` array whose first element has
`message` (`role` = `assistant`, `content` as text or null, and `tool_calls` when
present) and a `finish_reason` of `stop`, `length`, or `tool_calls`. The response
SHALL include a `usage` object with `prompt_tokens` and `completion_tokens`.

#### Scenario: Plain answer
- **WHEN** the model answers with text and stops naturally
- **THEN** `choices[0].message.content` holds the text, there is no `tool_calls`, and `finish_reason` is `stop`

#### Scenario: Token limit reached
- **WHEN** generation is cut off by `max_tokens`
- **THEN** `finish_reason` is `length` and `content` holds the text generated so far

### Requirement: Tool declaration and tool calls
A request MAY include `tools`, each a function with `name`, `description`, and a
JSON-schema `parameters` object. When the model decides to call tools, the
response message SHALL carry `tool_calls` (each with a unique `id`, `type` =
`function`, `function.name`, and `function.arguments` as a JSON-encoded string),
`finish_reason` SHALL be `tool_calls`, and `content` SHALL be null or the text
produced before the call.

#### Scenario: Model calls a tool
- **WHEN** the request declares a `calculator` tool and the model decides to use it
- **THEN** the response has one `tool_calls` entry named `calculator` with arguments parseable as JSON and `finish_reason` `tool_calls`

#### Scenario: Tool result continues the conversation
- **WHEN** a follow-up request appends the assistant message with `tool_calls` and a `tool` message with the matching `tool_call_id` and the tool output
- **THEN** the model answers using that output and the response has `finish_reason` `stop`

#### Scenario: Tool message without a matching call
- **WHEN** a `tool` message references a `tool_call_id` that no earlier assistant message produced
- **THEN** the response is 400

#### Scenario: Unparsable tool call
- **WHEN** the model emits tool-call markup whose JSON cannot be parsed
- **THEN** the markup is returned as ordinary `content` with `finish_reason` `stop` and is not silently dropped

### Requirement: Streaming response
When `stream` is true the response SHALL be `text/event-stream` with one
`data: <json>` event per chunk (`object` = `chat.completion.chunk`, text in
`choices[0].delta.content`) and a final `data: [DONE]` event. The last JSON chunk
SHALL carry the `finish_reason`; if tool calls were produced they SHALL be
delivered complete in the final chunk's `delta.tool_calls`, not as partial
fragments.

#### Scenario: Text streams as it is generated
- **WHEN** a client requests `stream: true` and the model produces `Hi` then ` there`
- **THEN** the client receives chunks with delta content `Hi` and ` there`, then a chunk with `finish_reason` `stop`, then `[DONE]`

#### Scenario: Tool call markup is not streamed as text
- **WHEN** the model begins a tool call while streaming
- **THEN** no part of the tool-call markup appears in any `delta.content`, and the complete call arrives in the final chunk with `finish_reason` `tool_calls`

#### Scenario: Error during streaming
- **WHEN** generation fails after streaming has started
- **THEN** the stream sends a final event whose data is a JSON object with an `error` field and ends without `[DONE]`

### Requirement: Stop sequences
A request MAY give `stop` as a string or a list of up to 4 strings. Generation
SHALL end as soon as the output would contain any stop string, and the stop
string and anything after it SHALL NOT appear in the returned or streamed text.

#### Scenario: Stop string trims output
- **WHEN** `stop` is `["END"]` and the model would produce `done END ignored`
- **THEN** the returned content is `done ` and `finish_reason` is `stop`

#### Scenario: Stop string split across tokens
- **WHEN** a stop string spans two generated tokens
- **THEN** it is still detected, and no partial stop text is streamed

### Requirement: Model listing
`GET /v1/models` SHALL return a list containing one entry whose `id` is the
identifier of the loaded model. The `model` field of chat requests SHALL be
accepted with any value and SHALL NOT select a different model.

#### Scenario: List the loaded model
- **WHEN** a client sends `GET /v1/models`
- **THEN** the response is 200 with `data` containing exactly one model with a non-empty `id`

### Requirement: Context length and failure handling
If the rendered prompt does not fit the model's context window, the server SHALL
respond 400 with an error whose `code` is `context_length_exceeded` and SHALL NOT
run generation. Model failures SHALL return 500 with a generic message that does
not expose internal error details.

#### Scenario: Prompt too long
- **WHEN** the conversation exceeds the model's context window
- **THEN** the response is 400 with `error.code` `context_length_exceeded`

#### Scenario: Internal failure
- **WHEN** the model returns an error before producing output
- **THEN** the response is 500 with a JSON `error.message` that does not contain internal details

### Requirement: Chat endpoints share the single-model lock
Chat requests SHALL be serialised with all other inference requests on the loaded
model and SHALL have no request timeout. Cancelling a streaming request SHALL stop
generation and release the model.

#### Scenario: Cancelled stream frees the model
- **WHEN** a client disconnects in the middle of a streaming chat response
- **THEN** generation stops and a subsequent request is served without waiting for the abandoned one to finish
