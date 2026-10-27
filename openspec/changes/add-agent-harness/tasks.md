# Tasks

Conventions for whoever implements this (human or agent):

- **Prerequisite:** `restructure-workspace` is implemented. Check group 1 first.
- Run every cargo/just command inside the Nix dev shell:
  `nix develop --command just check` etc. `cargo` is not on the plain PATH.
- **Do not commit or stage.** The repository owner makes all commits and uses
  `jj`. Use plain file operations; inspect with `jj st` / `jj diff`; never run
  `git add|rm|mv|commit`.
- `just check` must pass at the end of **every** group. No test may download model
  weights or hit the network beyond localhost; real-model checks are in group 10.
- Read `design.md` D1-D13 before starting; task text refers to it by number.
- Every group lands its own tests and docs; do not defer them.

## 1. Prerequisites, baseline, and candle upgrade

- [ ] 1.1 Confirm change 1 landed: `djinn-models` exists in the workspace, `cargo tree -p djinn-core | grep -E "candle|tokenizers|hf-hub"` prints nothing, `djinn-server` has no `/clip` route. Verify: all three hold; if not, stop and finish `restructure-workspace` first.
- [ ] 1.2 Run `nix develop --command just check`; record test count and coverage as the baseline. Verify: exit 0.
- [ ] 1.3 Upgrade candle: bump `candle-core`, `candle-nn`, `candle-transformers` in the workspace `Cargo.toml` to the current release (0.11.0 when planned; use `cargo info candle-transformers` for the latest; the version itself is not a requirement). Only `djinn-models` should need source changes; fix compile errors there. If `tokenizers` or `hf-hub` need a matching bump, do it. Verify: `nix develop --command just check` exits 0 and `cargo tree -d` shows no duplicate candle versions.
- [ ] 1.4 Check the hardware feature builds still compile: `nix develop --command cargo check -p djinn-models --features cuda` on this Linux machine. State explicitly that `--features mac` could not be verified here (it is Apple Silicon only) unless it was run on a Mac. Verify: the cuda check passes, or the failure is recorded with its cause. If cached weights exist for the existing quantized Mistral, run the server once and POST `/complete` to confirm generation still works after the upgrade; otherwise note that this was not run.

## 2. Shared chat wire types (`djinn-core`)

- [ ] 2.1 Add `djinn_core::chat` with the types in design D2 (request, message, role, tool spec, tool call, response, choice, finish reason, usage, stream chunk/delta, model list, error body), serde-only, `utoipa::ToSchema` behind the `openapi` feature. Follow the D2 details (`content: Option<String>`, `arguments` as a JSON-encoded string, `stop` as string-or-array, ignore unknown fields). Verify: `cargo test -p djinn-core chat` passes.
- [ ] 2.2 Add golden JSON tests: deserialize a minimal request, a request with tools, an assistant message with `tool_calls` and null content, a `tool` message; serialize a response and a stream chunk and compare to literal JSON; `stop` as `"x"` and as `["x","y"]`; an extra unknown field is ignored. Verify: tests pass; `cargo tree -p djinn-core` still shows no candle/tokenizers/hf-hub.
- [ ] 2.3 Add a validation function for requests (empty `messages`, unknown role is already a deserialize error, more than 4 stop strings, `tool` message whose `tool_call_id` matches no earlier assistant `tool_calls` entry) returning a typed error with a human-readable message. Verify: table-driven tests cover each rule from `specs/chat-completions-api` and pass.

## 3. Model runtime, part 1: pure logic (`djinn-models`, no weights)

- [ ] 3.1 Verify model files on the Hub before writing code (design D4): find the Qwen2.5-Instruct 1.5B (default) and 0.5B GGUF repos/files, check whether the quantization is a single file or split, find the tokenizer repo, and fetch `tokenizer_config.json`'s `chat_template`. Record repo ids, file names, sizes, and the exact chat template text in a new file `djinn-models/src/lm/qwen/NOTES.md` (committed by the owner, not you). Verify: the notes file exists and states which size/quant is the default and whether it is a single file.
- [ ] 3.2 Implement the stop/holdback matcher (design D5/D6): fed text fragments, returns text safe to emit, detects stop strings and the literal `<tool_call>` trigger, flushes held text on normal end. Verify: unit tests for stop at start/middle/end, split across fragments (every possible split of a sample string gives identical output), no stop present, multiple stop strings, and tool-call trigger detection; all pass.
- [ ] 3.3 Implement `ChatTemplate` trait and `Qwen25Template::render` (design D3) from the notes in 3.1: default system message, tools block, assistant tool calls, tool results as tool-response turns, consecutive tool messages merged, trailing opened assistant turn. Verify: golden-string tests for plain conversation, no-system, tools declared, tool-call round trip, and two consecutive tool results, each derived from the official Jinja template; all pass.
- [ ] 3.4 Implement `Qwen25Template::parse_output` (design D6): zero/one/many calls, preamble preserved, truncated block and invalid JSON treated as plain content, unique ids. Verify: table tests for each case in `specs/chat-model-runtime` ("Tool call extraction") pass.

## 4. Model runtime, part 2: Qwen loading and generation

- [ ] 4.1 Extend config (`djinn-core`): add `ModelArchitecture::Qwen25Instruct` (serde `qwen2_5_instruct`), optional `weights_file` / `tokenizer_repo` on `ModelSource::HuggingFaceHub` (default `None`, defaults from the architecture per D4); keep `QMistral` for now. Update `hf_repo_id` and its test. Verify: `cargo test -p djinn-core` passes; existing `configs/model/q_mistral.toml` still validates.
- [ ] 4.2 Add `configs/model/qwen2_5_instruct.toml` (with `#:schema` header) using the repo/file chosen in 3.1; run `just schema`. Verify: the file validates against `model.schema.json` (a `validate_and_load::<ModelConfig>` test reads it) and `just schema` leaves no further diff.
- [ ] 4.3 Implement loading for `Qwen25Instruct` in `djinn-models` with `candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf`, tokenizer from `tokenizer.json`; add a `Model::Qwen25Instruct` variant and the EOS/end-of-turn tokens (`<|im_end|>` plus the tokenizer's EOS). Mark network/GPU code `#[coverage(off)]` like the existing loader. Verify: `cargo check -p djinn-models` passes (loading itself is exercised in group 10).
- [ ] 4.4 Implement `generate` (design D5): tokenize the rendered prompt, error with `ContextLengthExceeded` if it exceeds the context, call the model's `clear_kv_cache()` and start at position 0, cap `max_tokens` to remaining context, sample with the existing logits-processor settings, feed decoded fragments through the matcher from 3.2, end on end-of-turn/stop/limit, and report `FinishReason`, `prompt_tokens`, `completion_tokens`. Keep the token-decoding logic separable so it can be unit-tested with a fake logits source where practical. Verify: `cargo test -p djinn-models` passes; any logic you could test without weights is tested.
- [ ] 4.5 Add the `ChatModel` trait and the Qwen implementation (render -> generate -> stream text events, then parse tool calls at the end per D6), plus a scripted `FakeChatModel` for tests. Verify: tests with the fake and a fake generator show: text streams incrementally, a tool call is never emitted as text, finish reason `tool_calls`/`stop`/`length` is reported, and `<tool_call>` split across fragments is still hidden.

## 5. Server chat endpoints (`djinn-server`)

- [ ] 5.1 Add `chat` module and routes `POST /v1/chat/completions`, `GET /v1/models` on the inference router (design D7); share the single loaded model with `/complete` (do not load it twice). Implement non-streaming responses per `specs/chat-completions-api` including `usage` and `finish_reason`. Verify: handler tests using `FakeChatModel` cover: minimal request 200, empty messages 400, unknown role 400, plain answer, `length`, tool call response shape, tool-result follow-up, tool message without matching call 400, `/v1/models` entry.
- [ ] 5.2 Implement streaming (`stream: true`): chunk objects, final chunk with `finish_reason` and complete `tool_calls`, `[DONE]`, error event on mid-stream failure. Verify: tests for text chunks then finish then `[DONE]`; tool call arrives complete in the final chunk and never in `delta.content`; error mid-stream produces an `error` object and no `[DONE]`.
- [ ] 5.3 Implement stop sequences end to end and error mapping (context exceeded -> 400 `context_length_exceeded`; model failure -> 500 generic body). Verify: tests for stop trimming, stop split across tokens, context error, internal error does not leak details.
- [ ] 5.4 Disconnect handling: stop generating and release the lock when the client goes away mid-stream. Verify: a test starts a streaming request against a fake model that yields slowly, drops the response body, and asserts a second request completes promptly and the fake observed cancellation.
- [ ] 5.5 Register routes/schemas in `openapi.rs`, document in `README.md` routes table. Verify: a test fetches `/api-doc/openapi.json` and finds `/v1/chat/completions` and `/v1/models`; `just check` passes.

## 6. Agent crate: scaffold and tools

- [ ] 6.1 Create `djinn-agent` (lib + `[[bin]] djinn-agent`) with the dependencies from design D8 and add it to the workspace. Add a guard (a `just check` step or an `xtask` subcommand, as done for `djinn-core` in change 1) that fails if `cargo tree -p djinn-agent` contains `candle-core`, `djinn-models`, or `djinn-server`. Verify: the guard passes; temporarily adding `djinn-models` makes it fail.
- [ ] 6.2 Implement `Tool` trait and `ToolRegistry` (design D8): register (duplicate names rejected), lookup, `specs()` producing wire `ToolSpec`s, listing for `/tools`. Verify: unit tests for duplicate rejection, specs content, unknown lookup.
- [ ] 6.3 Implement `calculator` (design D11). Verify: tests for the spec scenarios (precedence, parentheses, unary minus, decimals, division by zero, `2 ** 3`, `system("ls")`, empty, overlong, deep nesting, missing argument, integer formatting without `.0`).
- [ ] 6.4 Implement `current_time`. Verify: test parses the output as RFC 3339 and compares with the system clock within 5 seconds.
- [ ] 6.5 Implement `read_file` (design D11). Verify: tempdir-based tests for read ok, `../` escape, absolute path outside the root, symlink escape (Unix), directory, missing file (message does not reveal outside paths), binary file, file over 64 KiB truncated with note, and a path with `.` segments that stays inside.

## 7. Agent crate: backend and loop

- [ ] 7.1 Implement the `Backend` trait, `BackendEvent`, `ChatResult`, `BackendError` (connection / status / stream ended early / malformed), and a `ScriptedBackend` for tests. Verify: compile + a test that the scripted backend replays a scripted sequence.
- [ ] 7.2 Implement the SSE line parser (design D9) as a pure function/struct over byte chunks. Verify: tests for `\n` and `\r\n`, one event split across chunks, a UTF-8 code point split across chunks, several events in one chunk, comments and `event:` lines ignored, multiple `data:` lines joined, `[DONE]`, missing trailing newline.
- [ ] 7.3 Implement `OpenAiBackend` (reqwest, streaming, bearer token only when configured, default URL/model) with tool-call assembly and id generation (design D9). Verify: tests against a local fake server (axum on `127.0.0.1:0`) cover text streaming, fragmented tool-call arguments, complete tool call in one chunk, missing ids, 400 with error body, connection refused (message names the URL), stream cut before the final chunk, bearer header presence/absence.
- [ ] 7.4 Implement `Agent::run_turn` and `AgentEvent` (design D10) with step limit, tool timeout, output truncation with marker, failure-as-tool-result, history truncation on backend error, and `reset`. Verify: tests with `ScriptedBackend` and fake tools for every scenario in `specs/agent-loop` (direct answer, one tool then answer, two calls in one reply, results carry call ids, step limit with limit=2, unknown tool, bad arguments, tool timeout via a sleeping fake tool with a short limit, oversized result truncation, follow-up keeps history, reset, backend error mid-turn discards the turn, events order).

## 8. Agent crate: REPL and binary

- [ ] 8.1 Implement CLI flags and env fallbacks per `specs/agent-repl` (`clap` derive, `env = ...`), the default system prompt (design D12), and `--system <file>` loading with an early error. Verify: tests with `Cli::try_parse_from` cover flag-over-env precedence, defaults, bad values exiting with usage error, missing system file error naming the file.
- [ ] 8.2 Implement the REPL: prompt, blank-line skip, streaming text to stdout, tool activity to stderr, slash commands `/exit /quit /reset /tools /help` and unknown-command handling, EOF exit 0, Ctrl-C cancelling a turn and truncating history (design D10/D12), startup reachability warning that does not abort. Structure it over `AsyncBufRead`/writers so it can be tested with in-memory I/O. Verify: tests with `ScriptedBackend` and in-memory I/O for each scenario in `specs/agent-repl` that does not require a real signal; the Ctrl-C path is tested through the same cancellation function the signal handler calls.
- [ ] 8.3 Implement `--once` (answer only on stdout; non-zero exit on backend error or step limit). Verify: tests for success output/exit code and for unreachable backend (empty stdout, URL on stderr, non-zero).
- [ ] 8.4 Add `agent *args` recipe to `justfile` and a `djinn-agent` section to `README.md` with the djinn-server and Ollama invocations. Verify: `nix develop --command just agent --help` prints usage; README commands match the flags.

## 9. Retire the base model, configs, docs

- [ ] 9.1 Remove `ModelArchitecture::QMistral`, its runtime variant and loader, `configs/model/q_mistral.toml`; set `configs/server/test.toml` to `qwen2_5_instruct.toml`; update `djinn-cli` defaults that referenced the Mistral config (`DEFAULT_MODEL_CONFIG`, `single-run` architecture); run `just schema`. Verify: a config naming `q_mistral` fails validation with an error listing valid variants (add a test); `just check` passes; `grep -ri "q_mistral\|lmz/candle-mistral" .` finds nothing outside `openspec/`.
- [ ] 9.2 Update `CLAUDE.md` (Workspace: `djinn-models`, `djinn-agent`; dependency rule for the agent; commands: `just agent`; feature-flag ownership) and `README.md` (crate list, routes, model, agent). Verify: every command mentioned in the docs was run once successfully (or the docs say it requires downloaded weights).
- [ ] 9.3 Run the full gate: `nix develop --command just check`. Verify: exit 0; report test count and coverage against the group 1 baseline; coverage stays above the 30% gate.

## 10. End-to-end verification (manual; needs weights)

- [ ] 10.1 Start the server with the 1.5B config (`just server` or `cargo run -p djinn-cli -- server-config --name test`), then: `curl /health-check`; `curl /v1/models`; a non-streaming chat; a streaming chat (`-N`); a chat with the `calculator` tool declared that should trigger a call. Verify: each behaves per `specs/chat-completions-api`; record the model size/quant, device, tokens/sec, and any template or parsing problems found.
- [ ] 10.2 Run the agent against the djinn server: `just agent --once "What is (17 * 23) + 5? Use the calculator."`, then an interactive session asking for the time and to summarise a file under `--tool-root`. Verify: tools are called with sensible arguments and the final answers are correct; record failures with the model's raw output.
- [ ] 10.3 Run the agent against Ollama (`--base-url http://<host>:11434/v1 --model <tool-capable model>`) with the same prompts. Verify: it works without code changes (this validates the "any backend" requirement); record which model was used.
- [ ] 10.4 Press Ctrl-C during a long streaming answer in the REPL, then immediately ask another question. Verify: the prompt returns, the server logs show generation stopped, and the next request is served without waiting.
- [ ] 10.5 Repeat 10.2 with the 0.5B model and compare tool-use reliability to 1.5B (optionally with a larger model if one is at hand). Verify: results noted (informational; no code change required).

## Workflow follow-up

- Archive both changes (`openspec archive restructure-workspace`, then `openspec archive add-agent-harness`) after review.
- Follow-ups intentionally left out: `ratatui` TUI consuming `AgentEvent`; confirmation flow plus write/exec tools; context-window management; KV-cache reuse across requests; dense Qwen3 (0.6B/1.7B, needs `<think>` handling) as a second model; `rustyline` line editing.
