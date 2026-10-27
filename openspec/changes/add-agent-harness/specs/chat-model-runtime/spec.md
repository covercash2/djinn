# Spec Delta

## Purpose

Defines what the model layer provides for chat: a tool-capable instruct model,
a faithful chat prompt format, extraction of tool calls from model output, and
correct stopping. It makes the chat API possible and is independent of HTTP.

## ADDED Requirements

### Requirement: Supported chat model
The model layer SHALL support Qwen2.5-Instruct in quantized GGUF form as its chat
model, loadable from the Hugging Face Hub by configuration, with its tokenizer
taken from the corresponding original model repository. The default configuration
SHALL select a small model (about 2 billion parameters or fewer) so it loads and
answers quickly on a CPU-only developer machine.

#### Scenario: Model loads from config
- **WHEN** the server starts with `configs/model/qwen2_5_instruct.toml`
- **THEN** the weights and tokenizer are fetched (or found in the Hub cache) and the model becomes ready to serve

#### Scenario: Config selects the weights file
- **WHEN** the model config names a different GGUF file or tokenizer repository
- **THEN** that file and repository are used instead of the defaults

### Requirement: Base-only model is gone
The unmodified base Mistral model (`q_mistral`) SHALL NOT be a selectable model
variant, and its config template SHALL be removed. A config naming it SHALL fail
validation with an error that lists the valid variants.

#### Scenario: Old config is rejected clearly
- **WHEN** the server is started with a config whose `variant` is `q_mistral`
- **THEN** startup fails with a validation error naming the valid variants

### Requirement: Chat prompt rendering
Conversations SHALL be rendered to the model's prompt in the exact format the
model was trained on (ChatML role markers), including a default system message
when none is given, assistant tool calls, and tool results. Declared tools SHALL
be described to the model in the system section in the model's documented format.

#### Scenario: Plain conversation
- **WHEN** the messages are system "S" and user "U"
- **THEN** the rendered prompt is the system turn, the user turn, and an opened assistant turn, with the model's role markers

#### Scenario: Default system message
- **WHEN** no system message is supplied
- **THEN** the model's default system message is inserted

#### Scenario: Tools are described
- **WHEN** one tool named `calculator` with a JSON-schema parameter object is declared
- **THEN** the system section lists it by name, description, and parameters in the model's tool format

#### Scenario: Tool results are rendered
- **WHEN** an assistant message with a tool call is followed by a tool message
- **THEN** the call appears in the assistant turn and the result appears as a tool-response turn as the model format requires

### Requirement: Tool call extraction
Generated text SHALL be scanned for the model's tool-call markup. Each complete,
valid call SHALL yield a tool name and a JSON arguments object; several calls in
one reply SHALL all be returned in order. Text before the first call SHALL be
preserved as content. Markup that is incomplete or invalid SHALL be treated as
plain text.

#### Scenario: One call
- **WHEN** the output is `<tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}</tool_call>`
- **THEN** one call named `calculator` with arguments `{"expression":"2+2"}` is returned

#### Scenario: Two calls
- **WHEN** the output contains two valid tool-call blocks
- **THEN** two calls are returned in order with distinct ids

#### Scenario: Preamble kept
- **WHEN** the output is `Let me check.` followed by a valid tool call
- **THEN** content is `Let me check.` and one call is returned

#### Scenario: Truncated call
- **WHEN** generation ends in the middle of a tool-call block
- **THEN** no call is returned and the text is preserved as content

### Requirement: Stopping rules
Generation SHALL end on the model's end-of-turn token, on a caller-supplied stop
string, or on the maximum token count, and SHALL report which of these occurred.
The end-of-turn marker SHALL never appear in output. A caller-supplied token limit
SHALL be capped at the room remaining in the context window.

#### Scenario: Natural end
- **WHEN** the model emits its end-of-turn token
- **THEN** generation ends with reason `stop` and the marker is not in the text

#### Scenario: Token limit
- **WHEN** the token limit is reached first
- **THEN** generation ends with reason `length`

#### Scenario: Limit larger than the context
- **WHEN** `max_tokens` exceeds the remaining context
- **THEN** the effective limit is the remaining context and generation ends with reason `length` if it is reached

### Requirement: Requests are independent
Each generation SHALL start from the rendered prompt only; earlier requests SHALL
NOT influence its output through leftover model state.

#### Scenario: Same request twice
- **WHEN** an identical request with a fixed seed and zero temperature is run twice in sequence
- **THEN** both runs produce identical output
