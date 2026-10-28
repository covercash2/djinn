# Design

## Context

State of the code this plan was written against (all committed):

- `djinn-core::chat` holds the OpenAI-subset wire types shared by the server and
  the agent. `name`, `id`, `tool_call_id`, `model`, `arguments` (a JSON *string*),
  `object` (a literal) and the sampling numbers are all bare `String`/`f64`/`usize`.
  `validate_request` checks a few cross-field rules after deserialization.
- `djinn-core::lm` holds `ModelSource::HuggingFaceHub { revision, repo_id,
  weights_file, tokenizer_repo }` (strings) and `RunConfig` (`sample_len`,
  `temperature`, `top_p`, ... plain numbers). Configs are loaded by
  `config::validate_and_load`, which validates the TOML against the generated
  JSON Schema **and then** deserializes.
- `djinn-agent`: `Args { base_url: String, model: String, api_key: Option<String>,
  .. }` (`#[derive(Debug)]`), `Tool::name(&self) -> &str`, `AgentEvent`,
  `BackendError`.
- `djinn-cli server`: `ip: String` (parsed late with `IpAddr::parse_ascii`),
  `model_config: String` (a config *name* like `model/qwen2_5_instruct`; the code
  appends `.toml`), `save_config: Option<String>`.
- `djinn-core` must stay dependency-light (no candle/tokenizers/hf-hub; enforced
  by `just check-core-deps`); `djinn-agent` must not depend on candle,
  `djinn-models` or `djinn-server` (`just check-agent-deps`).
- `just check` (typos, both guards, fmt, clippy `-D warnings`, nextest with a
  30% line-coverage gate; currently ~95%) is the gate for every group below.

See `proposal.md` for motivation and `specs/` for the observable rules.

## Goals / Non-Goals

**Goals:**
- Each value kind in `specs/domain-values/spec.md` is its own type, impossible
  to construct invalid, and impossible to confuse with another kind.
- One mechanism for all of them, so adding the next one is a few lines.
- No change to any serialized form; constraints show up in the schemas.
- Land in small steps that each pass `just check` (the owner commits each step
  with `jj`; never stage or commit on their behalf).

**Non-Goals:**
- Typing free text (message content, tool output, generated text). See the
  exemption list in the `domain-values` spec.
- Typing token counts (`Usage`), indices, timestamps, `Duration`s, or the
  `seed` (any `u64` is valid).
- `repeat_penalty` / `repeat_last_n`: same pattern, follow-up if wanted.
- The server-generated completion `id` (`chatcmpl-N`) and `created` timestamp:
  nothing consumes them as anything but opaque text.
- `ollama-cli` (standalone crate) and the Qwen golden fixtures.
- A general-purpose validation framework or derive macro crate.

## Decisions

### D1. Where the types live: `djinn_core::values`

Shared, wire-facing types go in `djinn-core/src/values/` (`mod.rs` for the
machinery, then `tool.rs`, `model.rs`, `text.rs`, `sampling.rs`). `djinn-core`
already hosts the config surface and the chat wire types, and both the server
and the agent depend on it. Agent-only types (`BaseUrl`, `ApiKey`) live in
`djinn-agent` (`djinn-agent/src/settings.rs`) because `url` and the redaction
policy are agent concerns and core must stay light.

Alternative: a new `djinn-values` crate. Rejected: one more crate for ~400 lines,
and `djinn-core` is already "the shared dependency-light crate".

### D2. One small hand-written macro, no new crates

A private `macro_rules!` (`string_value!`) generates, from a name, a `KIND`
label, a validation function and schema hints (`pattern`, `min_length`,
`max_length`):

- the tuple struct (private field), `Clone, Debug, PartialEq, Eq, Hash,
  PartialOrd, Ord`;
- `new(impl Into<String>) -> Result<Self, ValueError>`, `from_static(&'static
  str) -> Self` (panics with a clear message; **only for literals**, e.g.
  built-in tool names and tests), `as_str()`, `into_string()`;
- `Display`, `FromStr` (so clap parses it with no extra code), `AsRef<str>`,
  `Borrow<str>`, `TryFrom<String>`, `TryFrom<&str>`, `From<Self> for String`;
- serde via `#[serde(try_from = "String", into = "String")]`, i.e. **validated
  on deserialization**, serialized as the bare string;
- `schemars::JsonSchema` (hand-written: `type: string` plus the hints) and, behind
  the `openapi` feature, `utoipa::ToSchema` with the same hints.

A second macro (`number_value!`) does the same for bounded numbers
(`Temperature`, `TopP`, `MaxTokens`) with `minimum`/`maximum` hints.

There is deliberately **no** `From<&str>`/`From<String>`: the conversion can fail,
and an infallible one would defeat the point.

Alternatives: the `nutype` crate (clean, but no `schemars`/`utoipa` support, so
we would hand-write those anyway and add a proc-macro dependency to core);
`validator`/`garde` (validate structs after deserialization, which is exactly the
late validation we are removing); hand-writing each type (about 80 lines each,
and they would drift).

### D3. One error type, informative and safe

```rust
pub struct ValueError { kind: &'static str, reason: String, value: Option<String> }
```

Displays as ``tool name `read file`: only ASCII letters, digits, `_` and `-` are
allowed`` or, for secrets and when `value` is `None`, as `API key: must not
contain whitespace`. `kind` is chosen to match the wire/CLI name where there is
one (`temperature`, `top_p`, `max_tokens`, `stop sequence`, `--ip`-style names
come from clap), so API errors "name the field". Non-secret values are echoed
truncated to 64 characters; secret types never echo.

### D4. Validate on deserialization; keep `validate_request` for cross-field rules

Per-value rules move into the types. `validate_request` keeps what a single
value cannot know: `messages` non-empty, and each `tool` message referencing an
*earlier* assistant call (its `ToolCallId`s are now typed, so the `HashSet<&str>`
becomes `HashSet<&ToolCallId>`). The "at most 4 stop strings" rule moves into
`StopSequences` (a `Vec<StopSequence>` newtype with a length bound; `StopSpec`
deserializes straight into it), and `MAX_STOP_SEQUENCES` is re-exported from
`values` so it stays referenced in one place.

A serde error from a validated type reaches the server's `Json` extractor as an
ordinary data error and is rendered with the existing error mapping (400 with
`error.message`). **Task 3.x verifies the status is 400 and the message contains
the kind** — do not assume it. For TOML configs, rules expressible in JSON Schema
(pattern/length/bounds) are reported by the existing schema validation step with
the key path; the rest (e.g. `..` segments) by the TOML deserializer with the key
and line.

### D5. `ToolArguments` keeps the exact text and the parsed object

```rust
pub struct ToolArguments { raw: String, object: serde_json::Map<String, Value> }
```

`new(raw)` parses once. `as_str()` returns the original bytes (used when the
assistant message is replayed to a model, which keeps prompts byte-stable),
`object()` returns the parsed map (used by tools), and an empty/whitespace-only
string is accepted with `object()` empty — OpenAI clients commonly send `""` for
a call with no arguments. Equality compares `raw`. Streaming deltas
(`FunctionCallDelta::arguments`) stay `Option<String>` because a partial JSON
fragment is not a `ToolArguments`; the agent's stream assembler constructs one
when the call completes and treats failure as a malformed response (this is what
the agent already does with invalid JSON, just typed).

### D6. Strictness towards other servers

`ToolName`/`ToolCallId`/`ToolArguments` are used in the shared response types, so
the agent parses a foreign server's tool calls through the same validation. A
name such as `functions.get_weather` (dot) would fail. Decision: **stay
strict**, the rules are OpenAI's own (`^[a-zA-Z0-9_-]{1,64}$`) and what Ollama,
llama.cpp and vLLM emit in practice; a violating call is reported as a malformed
response naming the bad value instead of being half-handled. Missing ids are
already generated by the backend (unchanged). If a real server needs more, the
fix is a single constant in `values/tool.rs`. This is checked against the local
Ollama in task 8.2.

Alternative: parse leniently (`RawToolName`) in the agent and convert at the
registry. Rejected: two types for one concept, and the lenient path hides
protocol errors.

### D7. Literal `object` fields become marker types

`ChatResponse.object`, `ChatChunk.object`, and the model list's `object` become
unit-like marker types (`ChatCompletionObject`, `ChatCompletionChunkObject`,
`ListObject`) that serialize to the literal and, on deserialize, accept only that
literal. The fields are `#[serde(default)]` so a server that omits `object` on a
chunk is still accepted (it happens in the wild), while a *wrong* value is
rejected. `OBJECT_CHAT_COMPLETION*` constants are removed.

### D8. Sampling parameters

`Temperature(f64)` (0..=2), `TopP(f64)` (>0, <=1), `MaxTokens(NonZeroUsize)`.
`PartialEq` only (floats). Constants like `DEFAULT_TEMPERATURE` become `const`
values built with a `const fn new_unchecked_const` that contains a
`const { assert!(..) }` check, so the defaults are proven valid at compile time
(spec: "Defaults are valid"). They are used in `ChatRequest`, `ChatParams`,
`GenParams`, and `RunConfig` (`sample_len: MaxTokens`, `temperature:
Temperature`, `top_p: Option<TopP>`). `clap` parses them through `FromStr`; the
`RunConfig` JSON Schema gets `minimum`/`maximum`/`exclusiveMinimum`. The wire
and TOML field names do not change.

### D9. Agent connection settings

- `BaseUrl` wraps `url::Url`, must be `http`/`https` with a host and no userinfo,
  query or fragment (so no credentials can hide in a logged URL), and drops a
  trailing `/`. It exposes `endpoint(&str) -> Url` (replacing string
  concatenation) and `Display`. `BackendError::Connection.url` becomes `BaseUrl`
  or `Url`, `Status.status` becomes `reqwest::StatusCode`.
- `ApiKey` holds visible ASCII only (so it is always a valid header value) and has a
  hand-written `Debug` printing `ApiKey(<redacted>)`, no `Display`, no `Serialize`, and
  `expose() -> &str` used in exactly one place (the `Authorization` header, which is also
  marked `set_sensitive(true)` so `reqwest`'s own debug output hides it). `Args` keeps
  `#[derive(Debug)]` safely.
- clap normally prints a rejected value in its error message, which for a secret is exactly
  what must not happen, so `--api-key` uses a custom `TypedValueParser` (`ApiKeyParser`) that
  reports the rule without the value. Consequence: an *empty* `DJINN_AGENT_API_KEY` is now an
  error (it used to be silently ignored).
- `BaseUrl` does not echo a rejected URL that contains credentials (the value *is* the secret).
- `BackendError::Status` now holds a `reqwest::StatusCode`, so messages read
  `HTTP 500 Internal Server Error: ...` instead of `HTTP 500: ...`.
- The agent's `--model` is a `ModelId`, `--system` content a `SystemPrompt`.

### D10. CLI arguments

`ServerArgs.ip: IpAddr` (the late `parse_ascii` call disappears),
`model_config: ConfigName`, `save_config: Option<ConfigName>`; `single-run`'s
`weight_files`/`tokenizer_file` become `PathBuf`, `revision` a `Revision`, its
repo/model ids `RepoId`/`ModelId`. The default `"model/qwen2_5_instruct"` is built
with `ConfigName::from_static`. `server-config --name` and `config --name` take a
`ConfigName`. clap's own error text is already a usage error naming the flag.

### D11. Test ergonomics and fixtures

~300 tests build these values from literals. Tests use `X::from_static("..")` (or
`"..".parse().unwrap()`). The golden Qwen fixtures and `testing.rs` mocks keep
their `String` text. The `ScriptedBackend` and fake servers construct typed
calls via `ToolCall::function(..)`, whose parameters become `ToolCallId`,
`ToolName`, `ToolArguments` (callers pass `from_static` values).

### D12. Order of work

Each numbered group in `tasks.md` leaves `just check` green and is a natural
commit. Order: standard types in the CLI (no new machinery) → the value machinery
and all new types, tested in isolation and **not yet used** → tool values
through wire types, server, agent → model/Hub values and config/schema →
text roles → sampling → agent connection settings → documentation and
verification. Groups after 2 are independent of each other and can be dropped or
reordered without breaking the rest.

### D13. Instructions

`CLAUDE.md` gets a "Conventions" section with the rule and the exemptions, and
`openspec/config.yaml` gets a `context:` entry so that future specs, designs and
tasks written by any agent follow it. Both are edited as part of drafting this
change (documentation only), with a sentence saying existing code is being
migrated; task 9.3 removes that sentence when the migration is done.

## Risks / Trade-offs

- **Foreign servers send values our rules reject** (names with dots, odd ids, wrong
  `object`) → D6/D7 choices, plus live checks against Ollama (`llama3.2`,
  `qwen2.5`) in task 8.2; the rules live in one constant each if they must relax.
- **Large mechanical diff** (all crates, ~300 tests) → small ordered groups,
  each green on its own; review per group; no behaviour changes mixed in.
- **Schema churn** (`configs/**/*.schema.json`, OpenAPI) → regenerate with
  `just schema` per group and review the diff; spec requires idempotence.
- **Error text regressions**: some existing tests assert exact messages (e.g.
  the unknown-variant listing) → keep those messages, add new ones alongside.
- **`serde(try_from)` plus `utoipa`/`schemars` derives do not mix** → hand-written
  schema impls in the macro (D2); covered by schema snapshot tests.
- **A panic in `from_static` in production code** → documented literal-only use;
  a test constructs every built-in tool name and every default so a typo fails
  `just check`, not a user.
- **Coverage gate**: new modules are heavily unit-tested (boundary scenarios
  from the spec map 1:1 to tests), so the 30% gate is not at risk.
