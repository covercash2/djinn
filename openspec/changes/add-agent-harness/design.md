# Design

## Context

Read `proposal.md` (why/what) and `specs/` (behavior contracts) first. This file
is the *how*. **Prerequisite:** `restructure-workspace` is implemented; see the
note at the top of `proposal.md` for the exact assumptions about the tree.

Verified facts this design relies on (checked against source in the dev
environment, not assumed):

- The candle version is not a requirement. The workspace is on 0.8.4 today; the
  current release when this was planned is **0.11.0**, and this change upgrades to it
  first (group 1) while candle is confined to `djinn-models`. Checked in 0.11.0's
  source: `models::quantized_qwen2` and `models::quantized_qwen3` (dense) both
  exist, each `ModelWeights` has `from_gguf(...)`, `forward(&mut self, x, index_pos)`
  and an explicit `clear_kv_cache()`. (0.8.4 only resets the cache implicitly when
  `index_pos == 0`; do not depend on that.) Re-check the API against whatever
  version is actually resolved.
- `reqwest` 0.12 is already in `Cargo.lock` (via hf-hub). The server today serves
  `/complete` and `/complete/stream` from a single `Arc<Mutex<Box<dyn
  LanguageModel>>>`; streaming uses a `tokio::sync::mpsc` channel plus
  `Sse::new(ReceiverStream)`, and a dropped client is detected by a failed
  `tx.send`.
- `RunConfig` (sampling params: `sample_len`, `seed`, `repeat_last_n`,
  `repeat_penalty`, `temperature`, `top_p`) lives in `djinn-core` and is part of
  the `/complete` request body. Leave it as is.
- `prompts/system/help_chris.txt` exists and is a good `--system` example.
- The user makes all commits and uses `jj`. Implementers must not stage or
  commit; use plain file operations and `jj st` / `jj diff` for inspection.
- `cargo` is only available via `nix develop --command ...`.

## Goals / Non-Goals

**Goals:**

- A working agent: `djinn-agent` answers questions and uses `current_time`,
  `read_file`, `calculator`, against either the djinn server or Ollama.
- One wire protocol (OpenAI-compatible chat completions subset) used by server and
  agent.
- Everything testable without model weights: template rendering, tool-call
  parsing, stop handling, the loop, the HTTP backend (against a local fake
  server), tools.

**Non-Goals:**

- No TUI (`ratatui` comes later; the loop emits events so a TUI can plug in).
- No tools that write, execute, or use the network; no confirmation UI.
- No auth on the server, no multi-user sessions, no persistence of history.
- No context-window management (summarising/truncating). The server returns
  `context_length_exceeded`; the agent shows it.
- No cross-request KV-cache reuse (prefill is recomputed each request).
- No model-parallelism, batching, or more than one loaded model.
- No Jinja chat-template engine, no models other than Qwen2.5-Instruct.
- No models other than Qwen2.5-Instruct yet. Qwen3 (dense, available in the upgraded
  candle) is the likely second model but needs its own chat template and `<think>`
  handling; leave it for a follow-up.
- No changes to `ollama-cli` or to the `/complete` routes.

## Decisions

### D1. Wire protocol: OpenAI-compatible chat completions (subset)

The server implements `POST /v1/chat/completions` and `GET /v1/models`; the agent
speaks only that. Same protocol is served by Ollama (`/v1`), llama.cpp server and
vLLM, so "connect to whatever backend" is true from day one with a single HTTP
backend implementation.

Alternatives: (a) a bespoke djinn protocol: rejected, needs an adapter per other
backend and nothing else speaks it. (b) Ollama's native `/api/chat`: rejected as
the *only* protocol (couples to one product), but the `Backend` trait leaves room
to add it later.

Subset boundaries are in the spec: ignored unknown fields, no `n>1`, no `logprobs`,
`tool_choice` is implicitly `auto`, `model` is accepted and ignored.

### D2. Shared wire types live in `djinn-core`

New module `djinn_core::chat` with serde types: `ChatRequest`, `ChatMessage`,
`Role`, `ToolSpec`/`FunctionSpec`, `ToolCall`/`FunctionCall`, `ChatResponse`,
`Choice`, `FinishReason`, `Usage`, `ChatChunk`/`Delta`, `ModelList`, `ErrorBody`.
Derive `utoipa::ToSchema` behind the existing `openapi` feature (same pattern as
`RunConfig`). Both `djinn-server` and `djinn-agent` use these. `djinn-core` is
candle-free after change 1, so the agent depending on it pulls in nothing heavy
(verify with `cargo tree -p djinn-agent`).

Alternative: duplicate the types in the agent. Rejected: two copies of the same
protocol will drift and the golden JSON tests would have to be written twice.

Details to get right in the types: `content` is `Option<String>` (null when the
message only carries tool calls); `function.arguments` is a **string** containing
JSON (OpenAI convention), not an object; `stop` accepts a string or an array
(`#[serde(untagged)]`); unknown request fields are ignored (do not
`deny_unknown_fields`).

### D3. Chat templating and tool parsing live with the model, hand-written

In `djinn-models`: a `ChatTemplate` trait (`render(&[ChatMessage], &[ToolSpec])
-> String`, `parse_output(&str) -> ParsedReply`) with one implementation,
`Qwen25Template`. Rendering is plain Rust string building, covered by golden-string
tests. Tools are described in Qwen's documented format: a system section containing
a `<tools>` block with one JSON function spec per line and an instruction to reply
with `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`; tool results are
rendered as a user-role turn wrapped in `<tool_response>` tags (consecutive tool
messages are merged into one turn).

**First task of the template work: copy the exact format from the model's
`tokenizer_config.json` `chat_template` on the Hub**, and write the golden tests
from it. Do not rely on this document's description of the format; the Jinja
template is authoritative. If a detail differs (whitespace, default system
message text, handling of consecutive tool messages), follow the Jinja template.

Alternative: load `chat_template` from `tokenizer_config.json` and render it with
`minijinja`. More general, but pulls a template engine in, and tool-calling templates
use filters/macros that are easy to get subtly wrong. Revisit when a second model
family is added.

### D4. Model: small Qwen2.5-Instruct GGUF (1.5B default, 0.5B for fast checks)

Chosen because it is open, instruct-tuned, trained for Hermes-style tool calls, runs
via `quantized_qwen2`, and comes in small sizes. We are testing, not shipping, so
the default is deliberately small: **1.5B Q4_K_M** (roughly 1 GB, fast on a CPU-only
machine), with **0.5B** documented as the quick smoke-test option. Expect weak
multi-step tool use at these sizes; that is acceptable for now and is a reason the
loop has a step limit and feeds errors back. Larger models are a config change, not
a code change (the architecture is "Qwen2.5-Instruct", the size is just which
GGUF file is named).

`ModelSource::HuggingFaceHub` gains optional `weights_file` and `tokenizer_repo`
(serde default `None`; defaults derived from the architecture, so existing TOML
keeps parsing). GGUF weights come from a GGUF repo; the tokenizer
(`tokenizer.json`) comes from the original `Qwen/Qwen2.5-*-Instruct` repo.

**Verify before coding (first task of group 3):** exact repo ids and file names
on the Hub. Known risk: official Qwen GGUF repos **split larger quantizations
into multiple files** (e.g. `-00001-of-00002.gguf`), and candle's loader reads a
single file. For the small sizes a single file should exist; for larger sizes you may need a
community single-file GGUF repo. Write the chosen ids into the config template and the
README, and record which size/quant was actually tested.

Alternatives: Qwen3 small dense models (0.6B/1.7B) via `quantized_qwen3` (modern and
small, but emits `<think>` reasoning blocks that must be stripped/handled; planned
as the second template); Llama 3.x via `quantized_llama` (different template, weaker
small-model tool calling); Mistral-7B-Instruct (older, tool format varies by
version). Keep the `ModelArchitecture`/`ChatTemplate`
split so these can be added later.

### D4b. Amendments found while implementing groups 3-4

- **`ModelSource::HuggingFaceHub` also gets `repo_id`** (besides `weights_file` and `tokenizer_repo`), because choosing 1.5B vs 0.5B means choosing a different GGUF repository. All three are optional with architecture defaults, so old configs still parse. Two configs ship: `qwen2_5_instruct.toml` (1.5B) and `qwen2_5_instruct_0_5b.toml`.
- **Chat is a capability of `LanguageModel`, not a separate trait object.** `LanguageModel` gained `chat()` (default: `Error::ChatUnsupported`) and `model_id()`. The server keeps its single `Arc<Mutex<Box<dyn LanguageModel>>>`, so `/complete` and the chat routes share one loaded model with no second load (this resolves the "one trait or two" open question). The scripted test model is `djinn_models::testing::ScriptedModel` behind the `testing` feature.
- **Generation is split in three tested layers:** `generate_stream` (token loop over a small `StepModel` trait; tested with a scripted model and a word-level tokenizer, no weights), `reply_stream` (stop strings, tool-call hold-back and extraction over a `GenEvent` stream; tested without any model), and `ModelContext::chat` (wiring, needs weights). `Holdback` is the shared trigger matcher. Prompt tokens are encoded without added special tokens because the template already contains them, and the prompt is not echoed.
- **Tool-call ids** come from a process-wide counter (`call_<n>`), unique across responses so a long conversation never reuses an id.
- **`serde_json` now has `preserve_order`** workspace-wide so tool-parameter key order reaches the prompt exactly as the client sent it (Python's dict order, which the template preserves). Verified that the generated schema files are unaffected.
- **Logits shape:** Qwen's `forward` returns `(batch, vocab)` while Mistral's returns `(batch, 1, vocab)`; the shared code flattens the batch-of-one instead of squeezing a fixed number of dimensions.
- **Fixtures come from the real template:** `djinn-models/src/lm/qwen/fixtures.json` was generated by rendering the official Jinja template with Jinja2 (see `NOTES.md`), not hand-written.
- **Overlapping stop strings** (one a prefix of another) may report either, depending on chunking; documented on `Holdback::new`.

### D5. Generation core: one streaming function with stop handling

Today's `LanguageModel::run(prompt, RunConfig) -> Stream<String>` stays for
`/complete`. Add a second capability to the model context (not a replacement):

```
generate(prompt: String, params: GenParams) -> Stream<Result<GenEvent>>
GenEvent::Text(String) | GenEvent::Done { finish: FinishReason, prompt_tokens, completion_tokens }
GenParams { max_tokens, sampling (reuse RunConfig fields), stop: Vec<String> }
```

Stop-sequence handling is a separate pure type (`StopMatcher`) fed decoded text
fragments; it holds back up to `longest_stop - 1` trailing characters so a stop
string split across tokens is never partially emitted, and flushes the held text
on normal end. It is unit-tested without a model. The end-of-turn token
(`<|im_end|>`) and EOS are always stop conditions and never appear in output.
Before each generation call the model's `clear_kv_cache()` and start at position 0, so no
state leaks between requests (spec: "Requests are independent").
`max_tokens` is capped to `context_len - prompt_tokens`; if the prompt alone
exceeds the context, return a typed `ContextLengthExceeded` error *before*
generating (the server maps it to 400).

Build the testable seam first: a `ChatModel` trait in `djinn-models`
(`fn chat(&mut self, ChatInput) -> Pin<Box<dyn Stream<Item = Result<ChatEvent>>>>`)
where the Qwen implementation does template rendering + generation + tool parsing,
and a scripted fake implements it for server tests (mirrors how the server tests
today use a mock `LanguageModel`).

### D6. Tool-call extraction and streaming

Parsing rules (see spec `chat-model-runtime`): scan for `<tool_call>` ...
`</tool_call>`; JSON inside must have string `name` and object `arguments`; ids are
generated (`call_<n>` unique per response). Invalid or truncated blocks are
returned as plain content (never dropped silently).

Streaming: stream text as it is generated, but hold back any suffix that could be
the beginning of the literal `<tool_call>` (same holdback mechanism as stop
strings; implement them in one `Holdback`/matcher utility with two kinds of
triggers). When `<tool_call>` is seen, stop emitting content, keep buffering until
generation ends, parse, and deliver complete tool calls in the final chunk. Do not
attempt to stream partial arguments.

### D7. Server routes and plumbing

New `djinn-server/src/chat/mod.rs` with `ROUTE_CHAT = "/v1/chat/completions"` and
`ROUTE_MODELS = "/v1/models"`, added to the *inference* router (no timeout; same
mutex). `AppState` holds the chat-capable model handle. Since both `/complete`
and chat need the same loaded weights, both go through the same
`Arc<Mutex<...>>`: **do not load the model twice**. Extend the existing trait
object, or hold a struct implementing both capabilities; pick whichever is the
smaller diff and note it in the PR/commit description. Streaming reuses the
mpsc + `Sse` pattern and its disconnect detection (`tx.send(...).is_err()` ->
stop generating and drop the lock). Add the routes to `openapi.rs`. Error
mapping: invalid request -> 400 JSON `{"error":{"message","type","code"}}`;
context exceeded -> 400 with `code: "context_length_exceeded"`; model error -> 500
generic message (as today).

Validation lives in one function (`validate_chat_request`) with table tests:
empty messages, unknown role, tool message without matching prior `tool_call_id`,
more than 4 stop strings.

### D8. `djinn-agent` crate layout

```
djinn-agent/
  Cargo.toml           # lib + [[bin]] djinn-agent
  src/lib.rs           # re-exports
  src/backend/mod.rs   # trait Backend, BackendError, ChatResult
  src/backend/openai.rs# OpenAiBackend (reqwest + SSE parsing)
  src/backend/scripted.rs  # ScriptedBackend for tests (cfg(test) or `testing` module)
  src/tools/mod.rs     # trait Tool, ToolRegistry, ToolError
  src/tools/{current_time,read_file,calculator}.rs
  src/agent.rs         # Agent, AgentConfig, run_turn(), AgentEvent
  src/repl.rs          # line loop, slash commands, event printing
  src/main.rs          # clap args -> Agent -> repl / --once
```

Dependencies: `djinn-core` (wire types only), `reqwest` (`json`, `stream`,
default TLS feature set chosen so `cargo tree` shows no *new* TLS stack),
`tokio`, `tokio-stream`/`futures`, `serde`, `serde_json`, `clap`, `thiserror`,
`anyhow` (binary only), a small time library for RFC 3339 (`time` with `formatting`,
or `jiff`; pick the smaller and justify). **Not** allowed: candle, `djinn-models`,
`djinn-server` (enforced by task 6.1).

`Backend` trait (async, object-safe via `async_trait`-free boxing, or generic over
`B: Backend`; prefer generics + `dyn` only at the binary edge to keep it simple):

```
trait Backend { fn chat(&self, req: ChatRequest) -> BoxStream<'_, Result<BackendEvent, BackendError>>; }
enum BackendEvent { Text(String), Done(ChatResult) }
struct ChatResult { content: Option<String>, tool_calls: Vec<ToolCall>, finish: FinishReason }
```

`Tool` trait: `name() -> &str`, `description() -> &str`, `parameters() ->
serde_json::Value` (JSON schema), `async fn call(&self, args: Value) -> Result<String,
ToolError>`. `ToolRegistry::register` rejects duplicate names; `specs()` yields
`Vec<ToolSpec>` for requests.

### D9. SSE parsing in the backend

Hand-written, line-based parser over `reqwest`'s byte stream: buffer bytes, split
on `\n`, ignore comment lines and `event:` names, concatenate multiple `data:`
lines, stop at `data: [DONE]`. Handle `\r\n`, chunk boundaries that split a line or a
UTF-8 code point, and a missing trailing newline. Reuse the same test-vector style as
the old `app.js` tests (chunked input, multiple events per chunk, split events).
No `eventsource` crate: it is ~40 lines and keeps dependencies down.

Tool-call assembly: accumulate `delta.tool_calls[i]` by `index`, concatenating
`function.arguments` fragments; parse once at the end. Some servers (including ours,
by spec) send complete calls in one chunk; both paths share the accumulator.
Generate ids (`call_<n>`) when absent. Arguments that fail to parse as JSON ->
`BackendError::MalformedResponse` (the loop turns it into a turn error).

### D10. Agent loop (`Agent::run_turn`)

Pseudo-algorithm (the spec scenarios are the test list):

```
history.push(user message)
for step in 1..=max_steps:
    stream = backend.chat(history, tools.specs(), opts)
    forward Text events to the event sink; collect ChatResult
    if result.tool_calls is empty:
        history.push(assistant(content)); emit TurnFinished; return Ok(answer)
    history.push(assistant(content, tool_calls))
    for call in result.tool_calls (in order):
        emit ToolStarted; run with tokio::time::timeout(tool_timeout)
        result text or error text -> truncate to max_tool_output with marker
        history.push(tool message { tool_call_id, content })
        emit ToolFinished
on max_steps: emit step-limit notice; return Err(StepLimit); history keeps what happened
on BackendError: truncate history back to the length before this turn; return Err
```

`AgentEvent`: `Text(String)`, `ToolStarted{id,name,args}`, `ToolFinished{id,
ok, summary, elapsed}`, `StepLimit`, `TurnFinished`. The REPL consumes events via
a channel or callback; a future TUI does the same.

Cancellation: `run_turn` is a future; the REPL races it against Ctrl-C
(`tokio::select!` with `tokio::signal::ctrl_c`) and drops it, then truncates history
to the pre-turn length. Dropping the future drops the in-flight HTTP request, which
closes the connection so the server stops generating (that relies on D7's
disconnect detection; test it in task 5.4 and end to end in task 10.4).

### D11. Built-in tools

- `current_time`: UTC, RFC 3339, seconds precision.
- `calculator`: hand-written recursive-descent parser/evaluator over `f64`; grammar
  `expr = term (('+'|'-') term)*`, `term = unary (('*'|'/') unary)*`,
  `unary = '-' unary | primary`, `primary = number | '(' expr ')'`. No variables,
  functions, exponent operator, or implicit multiplication. Format results without
  a trailing `.0` for integers. Reject non-finite results. Cap expression length
  (e.g. 200 chars) and nesting depth (e.g. 50) so it cannot be made to recurse deeply.
- `read_file`: `root` is canonicalised once at startup. For each call: join with the
  root (reject absolute paths unless they are inside the root after
  canonicalisation), `std::fs::canonicalize` the target (this resolves symlinks and
  `..`), require `target.starts_with(root)`, require a regular file, read at most 64
  KiB (+1 byte to detect truncation), `String::from_utf8` or error. Error messages
  must not echo paths outside the root. Use `tokio::task::spawn_blocking` or
  `tokio::fs`. Known limitation: a TOCTOU race between check and read is not
  defended against (read-only tool, local use); note it in the doc comment.

### D12. REPL and CLI

Plain stdin/stdout: `tokio::io::BufReader(stdin).lines()` racing Ctrl-C. Assistant
text -> stdout; tool activity, warnings, errors -> stderr (so `--once` output is
pipe-friendly). No line editing/history (`rustyline` is a cheap later upgrade, and
the TUI will replace this). Flags/env per the `agent-repl` spec using `clap` derive
with `env = ...`. Default system prompt: a short built-in constant that tells the
model it is a concise assistant, names no tools (they are provided via the API), and
says to use tools when helpful and to answer directly otherwise. `--system <file>`
replaces it. Exit codes: 0 ok, 1 runtime failure, 2 usage error (clap default).

### D13. Config and docs

New `configs/model/qwen2_5_instruct.toml` (+ schema header) as the default;
`configs/server/test.toml` points at it; delete `configs/model/q_mistral.toml`;
re-run `just schema`. New `justfile` recipe `agent *args` running `cargo run -p
djinn-agent -- {{args}}`. `CLAUDE.md`: add `djinn-agent` to Workspace, add the
dependency rule (agent must not depend on candle/models/server), add the agent
commands. `README.md`: a short "Agent" section with the Ollama and djinn-server
invocations.

## Risks / Trade-offs

- **Small quantized models are unreliable at tool use** (0.5B-1.5B especially) →
  keep tools few and their descriptions explicit; step limit; errors fed back; the
  manual tests (group 10) are run on 1.5B and 0.5B and results recorded; a
  larger model is an optional comparison. Quality is a property of the model, not a defect of the harness; do not
  paper over it in code.
- **Chat-template fidelity** (a subtle whitespace difference degrades quality
  silently) → golden tests derived from the official Jinja template (D3).
- **GGUF split files** (D4) → verify repo/file ids first; fall back to a
  single-file community repo and record it.
- **Streaming holdback bugs** (partial `<tool_call>` leaking into text, or text
  swallowed) → one tested holdback utility; property-style tests that feed the same
  text in every possible chunking and assert identical output.
- **Single-model mutex**: an agent turn with several model calls interleaves with
  other clients between calls (lock is per request) → acceptable; documented.
- **Prefill cost grows with the conversation** (no KV reuse) and the context
  window is finite → surfaced as `context_length_exceeded`; `/reset` is the user's
  escape hatch; non-goal to fix now.
  Measured during apply (CPU, 1.5B Q4_K_M, release): prompt processing ~57 tokens/s, so the agent's ~350-token system+tools prefix costs ~6 s before the first token of every turn, and an abandoned request keeps the server busy until its current prefill chunk ends. Reusing the KV cache of the unchanged prefix across requests (and/or processing the prompt in interruptible chunks) is the highest-value follow-up.
- **Prompt injection through `read_file` content** → read-only tools only, root
  sandbox; noted in the `agent-tools` spec as the reason new capabilities need their
  own change with confirmation.
- **Disconnect propagation** (agent Ctrl-C actually stopping generation) depends on
  D7's `send` failure check running between tokens → covered by an explicit
  test/manual check.
- **Candle upgrade (0.8.4 -> current)** can change APIs and transitive dependency
  versions (cudarc, half, tokenizers/hf-hub compatibility) and the CUDA/Metal builds
  are not exercised by `just check` → do the upgrade first, in its own tasks (1.3-1.4),
  with the Linux CUDA-feature compile check; the macOS `mac` feature cannot be
  verified on this machine, so say so rather than claiming it works.
- **Removing `q_mistral`** breaks anyone's local config naming it → the spec
  requires a clear validation error listing valid variants.
- **Big downloads in CI** → no test may download weights. Everything runs on
  fakes; real-model checks are manual tasks.

## Migration Plan

1. Land group order as in `tasks.md` (types -> model runtime -> server -> agent ->
   cleanup of Mistral -> docs). Each group leaves `just check` green.
2. Users with `configs/model/q_mistral.toml` copy `qwen2_5_instruct.toml`
   instead. `/complete` clients keep working (now backed by the instruct model;
   output style differs because there is no chat template on that route).
3. Rollback: revert the working-copy changes via the VCS; no persistent data.

## Open Questions

- Which exact Qwen2.5 GGUF repo and file to default to (decided at the start of
  group 3 after checking the Hub; does not change specs).
- Whether `ChatModel` and `LanguageModel` should be one trait or two behind one
  struct (implementer's choice per D7; both satisfy the specs).
- Whether the second model should be dense Qwen3 (0.6B/1.7B) once the REPL works
  (needs `<think>` handling; separate follow-up change).
- Whether to add `rustyline` for line editing now (deferred; TUI is planned).
