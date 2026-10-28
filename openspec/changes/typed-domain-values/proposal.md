# Proposal

> **Sequence:** independent of other changes; it builds on the archived
> `restructure-workspace` and `add-agent-harness` changes (all of which are
> already implemented). It touches every crate but changes **no wire format**.

## Why

Identifiers, paths, addresses, URLs, secrets and tuning parameters are all
passed around as bare `String` (and `f64`/`usize`). Code review found, for
example, that:

- `ToolCall::id`, `FunctionCall::name`, `ChatMessage::tool_call_id`,
  `ChatResponse::model`, `ModelSource::revision` and `repo_id` are all just
  `String`, so swapping two of them compiles and is only caught (maybe) by a
  test.
- Rules that exist in the specs are enforced late, or in several places, or not
  at all: stop strings are counted in `validate_request` but an **empty** stop
  string is accepted; a tool name may contain spaces; `--ip` is parsed deep in
  the server start-up, not when the arguments are parsed; a negative
  `temperature` or a `top_p` of 7 reaches the sampler.
- The agent's API key is a plain `Option<String>` inside a `#[derive(Debug)]`
  struct, so a stray `{args:?}` prints the secret.

Making each of these its own validated type means an invalid value cannot exist
past the place it enters the program (an HTTP body, a TOML file, a command-line
argument), and the compiler tells us what each value is.

## What Changes

- **New `djinn_core::values` module** with small validated newtypes. Each one is
  checked in its constructor and in `Deserialize` (so bad input is rejected at
  the HTTP/TOML/CLI boundary), serializes exactly like the `String`/number it
  wraps (`#[serde(transparent)]` semantics, so **no wire or file-format
  change**), and publishes its constraints in the generated JSON Schema and
  OpenAPI document:
  - tool values: `ToolName`, `ToolCallId`, `ToolArguments`;
  - model and Hub values: `ModelId`, `RepoId`, `Revision`, `WeightsFile`,
    `ConfigName`;
  - text roles: `Prompt`, `SystemPrompt`, `StopSequence` (and a
    `StopSequences` list that enforces the existing limit of 4);
  - sampling parameters: `Temperature`, `TopP`, `MaxTokens`.
- **In `djinn-agent`:** `BaseUrl` (http/https URL with a host) and `ApiKey`
  (non-empty, redacted `Debug`, no `Display`).
- **Standard types instead of strings:** `IpAddr` for `--ip`, `PathBuf` for
  file arguments, `StatusCode` for HTTP statuses, unit marker types instead of
  the literal-valued `object: String` fields.
- **Callers are migrated:** the wire types in `djinn_core::chat`, `ModelSource`,
  `RunConfig`, the server's handlers, the model layer, the agent's `Tool`
  trait, `AgentEvent`, backend and CLI arguments, and the `djinn-cli`
  arguments.
- **Instructions updated:** `CLAUDE.md` and `openspec/config.yaml` gain the rule
  "use a validated newtype for every domain value; `String` only for genuinely
  free text", with the list of intentional exemptions.
- **BREAKING (behaviour, not format):** inputs that used to be accepted are now
  rejected with 400 / a config error / a usage error: tool names with characters
  outside `[A-Za-z0-9_-]` or longer than 64, empty `tool_call_id`, empty stop
  strings, `temperature` outside `0..=2`, `top_p` outside `(0, 1]`,
  `max_tokens` of 0, malformed Hub `repo_id`/`revision`/`weights_file`, an
  unparsable `--ip` or `--base-url`, an empty API key or model name. Valid
  inputs behave exactly as before.

## Capabilities

### New Capabilities
- `domain-values`: the catalogue of validated value types, their validity
  rules, the guarantee that they validate on construction *and* on
  deserialization without changing the serialized form, how their constraints
  appear in schemas, and the list of values that intentionally remain free text.

### Modified Capabilities
- `chat-completions-api`: requests are rejected with 400 (naming the field)
  when a tool name, tool-call id, stop string, tool arguments, or sampling
  parameter is invalid.
- `inference-server-api`: `/complete` rejects out-of-range sampling parameters
  the same way.
- `chat-model-runtime`: model-source settings are validated when the config is
  loaded; extracted tool calls must carry a valid tool name.
- `agent-backend`: connection settings (base URL, model name, API key) are
  validated at start-up and the API key never appears in debug output; tool
  calls from servers with invalid names or ids are reported as malformed
  responses.
- `agent-tools`: the tool contract uses validated tool names.
- `djinn-cli-commands`: command-line values are validated when arguments are
  parsed.
- `workspace-layout`: the "domain values are validated newtypes" convention and
  its exemptions become a workspace rule.

## Impact

- **Code:** all crates. `djinn-core` gains `values/` (no new heavy
  dependencies; hand-written validation, no regex crate). `djinn-agent` gains
  the `url` crate (already in the tree through `reqwest`). Public function
  signatures that took `String`/`impl Into<String>` for these values change.
- **Generated files:** `configs/**/*.schema.json` gain `pattern`, `minLength`,
  `maxLength`, `minimum`/`maximum`; the OpenAPI document gains the same. Run
  `just schema` and commit the result.
- **Tests:** ~300 existing tests construct these values from string literals;
  most need `.parse()`/`::new(..).unwrap()` (a small `test-values` helper
  keeps that readable). Golden fixtures (`qwen/fixtures.json`) are unchanged.
- **Compatibility:** requests and configs that were valid and sane stay valid.
  The risks (foreign servers that send unusual tool-call names, empty-string
  arguments, or a wrong/missing `object`) are called out in `design.md` and
  checked against the local Ollama in the tasks.
