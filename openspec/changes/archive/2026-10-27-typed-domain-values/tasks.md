# Tasks

> Run everything through the Nix dev shell: `nix develop --command <cmd>` (cargo,
> just, typos exist only there). **Every group ends with `just check` green and
> `just schema` leaving the tree unchanged.** The owner commits with `jj`; do not
> stage or commit. Read `design.md` first — decisions D1–D13 are referenced below.
> Groups 3–7 are independent of each other once group 2 is done.

## 1. Standard types in the CLI (no new machinery)

- [x] 1.1 `djinn-cli/src/server.rs`: change `ServerArgs.ip` from `String` to `std::net::IpAddr` (default `::1`), delete the late `IpAddr::parse_ascii` call, and add a test that `--ip not-an-ip` is a clap error naming `--ip` while `--ip 127.0.0.1` and `::1` parse. Verify with `cargo nextest run -p djinn-cli`.
- [x] 1.2 `djinn-cli/src/single_run.rs`: change `weight_files` / `tokenizer_file` from `Option<String>` to `Option<PathBuf>`; adjust the consumers and add a parsing test. Verify with `cargo nextest run -p djinn-cli` and `cargo run -q -p djinn-cli -- single-run --help`.
- [x] 1.3 Update the README/CLI doc comments for the stricter `--ip` and verify `just check`.

## 2. The value machinery and all shared types (not yet used elsewhere)

- [x] 2.1 Create `djinn-core/src/values/mod.rs` with `ValueError` (D3), the private `string_value!` and `number_value!` macros (D2) generating the impl set listed in D2 (no `From<&str>`), and unit tests using a throwaway test-only type that cover: construction errors, `FromStr`, serde round trip of the bare string/number, rejection on `Deserialize`, `Display`, truncation of echoed values to 64 chars. Verify `cargo nextest run -p djinn-core values::`.
- [x] 2.2 `values/tool.rs`: `ToolName`, `ToolCallId`, `ToolArguments` (D5) per the `domain-values` requirement "Tool identifiers"; one test per scenario (valid/invalid names incl. 64/65 chars, arguments `{"expression":"2+2"}`/`{}`/``/`[1]`/`"x"`/`{"a":`, exact-text preservation). Verify `cargo nextest run -p djinn-core values::tool`.
- [x] 2.3 `values/model.rs`: `ModelId`, `RepoId`, `Revision`, `WeightsFile`, `ConfigName` per "Model and config identifiers" and "Hub identifiers"; tests for every scenario (path escapes, `a/b/c`, `owner/`, `test.toml` message). Verify `cargo nextest run -p djinn-core values::model`.
- [x] 2.4 `values/text.rs`: `Prompt` (any text), `SystemPrompt` (non-blank), `StopSequence` (1–256 bytes), `StopSequences` (≤ 4, re-exports `MAX_STOP_SEQUENCES`); tests per "Text roles". Verify `cargo nextest run -p djinn-core values::text`.
- [x] 2.5 `values/sampling.rs`: `Temperature`, `TopP`, `MaxTokens` with compile-time-checked `const` constructor for defaults (D8); tests for boundaries 0/0.7/2/-0.1/2.1/NaN/inf, `top_p` 0.01/1/0/1.5, token limit 0, and that `DEFAULT_TEMPERATURE` etc. build. Verify `cargo nextest run -p djinn-core values::sampling`.
- [x] 2.6 Schema output: add `insta` snapshot tests of `schemars::schema_for!` for each type (pattern/length/bounds present) and, under `--features openapi`, of the `utoipa` schema; verify with `cargo nextest run -p djinn-core --features openapi values::`.
- [x] 2.7 Verify `just check-core-deps` still passes (no new heavy dependency in `djinn-core`), then `just check`. Document the module in a `//!` header that lists each type, its rule, and the free-text exemptions.

## 3. Tool values through the wire types, runtime, server and agent

- [x] 3.1 `djinn-core/src/chat`: change `FunctionSpec.name`, `FunctionCall.name`, `FunctionCall.arguments`, `ToolCall.id`, `ChatMessage.tool_call_id` to `ToolName`/`ToolName`/`ToolArguments`/`ToolCallId`/`Option<ToolCallId>`; keep `FunctionCallDelta` fragments as strings (D5); replace the literal `object` strings with marker types and `#[serde(default)]` (D7), remove `OBJECT_CHAT_COMPLETION*`; update constructors (`ToolSpec::function`, `ToolCall::function`, `ChatMessage::tool`) and `validate.rs` (`HashSet<&ToolCallId>`). Update the module docs. Verify `cargo nextest run -p djinn-core chat::` including new tests for: invalid name/ids/arguments rejected on `serde_json::from_str`, `""` arguments accepted, absent `object` accepted, wrong `object` rejected.
- [x] 3.2 `djinn-models/src/lm/chat.rs`, `qwen/mod.rs`, `qwen/pyjson.rs`: `ParsedReply`/`next_call_id` produce typed values; an extracted call with an invalid name is treated as invalid markup and kept as text ("Extracted tool calls have valid names"). Verify the Qwen golden-fixture tests are unchanged and green (`cargo nextest run -p djinn-models qwen`) and add the invalid-name test.
- [x] 3.3 `djinn-server/src/chat/mod.rs`: adapt handlers/tests to the typed values. Add server tests for every scenario of "Request values are validated when parsed" that concerns tool names, tool-call ids and arguments. **Verify the HTTP status is 400 and the JSON `error.message` names the value kind** (design D4) — if the `Json` extractor yields 422 or a bare serde message, fix the mapping in `djinn-server/src/error.rs`. Verify `cargo nextest run -p djinn-server`.
- [x] 3.4 `djinn-agent`: `Tool::name(&self) -> &ToolName`, `DuplicateTool(ToolName)`, `AgentEvent::{ToolStarted, ToolFinished}`, backend stream assembler (builds `ToolArguments` when a call completes; failure → malformed response, "Malformed identifiers from servers are reported"), `ScriptedBackend`, built-in tools (`ToolName::from_static`). Add tests: bad tool name from a fake server → `BackendError` naming it, missing `object` accepted, ids generated when absent. Verify `cargo nextest run -p djinn-agent` and the 13 binary tests in `tests/cli.rs` still pass.
- [x] 3.5 Run `just schema` (OpenAPI/JSON Schema diff reviewed) and `just check`.

## 4. Model, Hub and config identifiers

- [x] 4.1 `djinn-core/src/lm/mod.rs`: `ModelSource::HuggingFaceHub { revision: Revision, repo_id: Option<RepoId>, weights_file: Option<WeightsFile>, tokenizer_repo: Option<RepoId> }`, `ModelSource::hub(Revision)`; `Files` already uses `PathBuf`. Update `ModelArchitecture::{hf_repo_id, default_weights_file, default_tokenizer_repo}` to return the typed values (via `from_static`). Add config tests for the three "Model source settings are validated at config load" scenarios and a test that every file under `configs/model/` still loads.
- [x] 4.2 `djinn-models`: `LanguageModel::model_id() -> ModelId`, `ModelContext`/loader/`testing.rs` mocks; derive the id from the weights file stem with a fallback that is always valid. `djinn-server`: `AppState.model_id`, `/v1/models`, `ChatResponse.model`. Verify `cargo nextest run -p djinn-models -p djinn-server`.
- [x] 4.3 `djinn-cli`: `ServerArgs.model_config` / `save_config` and the `server-config` / `config` `--name` become `ConfigName` (D10); `single-run` `revision`/repo/model ids become `Revision`/`RepoId`/`ModelId`. Add CLI tests for the three scenarios of "Command-line values are validated when parsed" (`--model-config ../elsewhere`, `--name test.toml`, documented examples still parse). Verify `cargo nextest run -p djinn-cli` and run the README/`justfile` commands' argument forms.
- [x] 4.4 Run `just schema`; review the `configs/**/*.schema.json` diff (patterns/lengths appear, nothing else moves) and confirm the "unknown variant lists valid variants" test still passes. `just check`.

## 5. Text roles

- [x] 5.1 `StopSpec` deserializes into `StopSequences` (string or list); `ChatRequest.stop: Option<StopSequences>`; `ChatParams.stop: Vec<StopSequence>`; `reply_stream`/`Holdback` take stop sequences as typed values where they are user-supplied (the internal tool-call trigger stays a `&str`). Remove the count check from `validate_request`. Add the "Empty stop string" and "Too many stop sequences" tests at core, model and server levels; the existing "stop string split across tokens" test must still pass.
- [x] 5.2 `/complete` and `/complete/stream` request/response `prompt` become `Prompt` (empty accepted — test it); `ModelRun.prompt`, `LanguageModel::run`, `single-run`'s prompt use `Prompt`. Verify the existing `/complete` tests unchanged and `cargo nextest run -p djinn-server complete`.
- [x] 5.3 Agent: the system prompt (built-in and `--system` file) is a `SystemPrompt`; an empty/blank `--system` file is an error naming the file ("Blank system prompt"). Add a binary test in `djinn-agent/tests/cli.rs`. `just schema && just check`.

## 6. Sampling parameters

- [x] 6.1 `ChatRequest.{temperature, top_p, max_tokens}`, `ChatParams`, `GenParams`, and `RunConfig.{sample_len, temperature, top_p}` take `MaxTokens`/`Temperature`/`TopP` (D8); `DEFAULT_*` constants are the compile-time-checked values; `RunConfig` clap args use `FromStr` and keep their env var names and defaults. Verify `cargo nextest run -p djinn-core -p djinn-models`, and that `djinn-cli single-run --temperature 3` is a usage error naming `--temperature`.
- [x] 6.2 Server tests for "Temperature out of range", "Zero token limit" on `/v1/chat/completions` and for all three scenarios of "Completion parameters are validated when parsed" on `/complete`; verify `/api-doc/openapi.json` shows `minimum: 0` / `maximum: 2` for `temperature` (a test). Remember the model layer caps the effective limit to the remaining context; that behaviour is unchanged.
- [x] 6.3 `just schema` (the `RunConfig` schema gains bounds; review) and `just check`.

## 7. Agent connection settings

- [x] 7.1 Add the `url` dependency to `djinn-agent`; create `djinn-agent/src/settings.rs` with `BaseUrl` and `ApiKey` (D9) and tests for every scenario of "Connection settings of the agent" (valid/invalid URLs, trailing slash dropped, key redaction in `Debug`, no `Display`). Verify `cargo nextest run -p djinn-agent settings::` and `just check-agent-deps`.
- [x] 7.2 Use them in `cli.rs` (`Args.base_url`, `Args.api_key`, `Args.model: ModelId`), `backend/openai.rs` (`endpoint()` replaces string concatenation; header marked sensitive), and `BackendError` (`Connection.url`, `Status.status: reqwest::StatusCode`). Add binary tests in `tests/cli.rs`: `--base-url localhost:8080` exits non-zero naming `--base-url` without sending a request; `--model ""` fails; with a key configured and a failing server, stdout/stderr never contain the key.
- [x] 7.3 `just check`.

## 8. Integration and live verification

- [x] 8.1 `just check` from a clean state, then `just schema` and confirm `jj diff configs/` shows no change after a second run (idempotence); `nix flake check --no-build` passes.
- [x] 8.2 Live checks (see the machine facts in the project memory: cached Qwen GGUFs and a local Ollama): (a) start the server with `configs/server/test.toml` and run `just agent -- --once "what is 12*7?"` — the answer uses the calculator; (b) run `djinn-agent --base-url http://localhost:11434/v1 --model llama3.2:latest` and `--model qwen2.5:latest` with a tool-using question — foreign-server tool calls must pass the strict validation (D6); (c) send `curl` requests for each 400 scenario in the specs and record the exact responses. If any real server is rejected, relax the single rule in `values/tool.rs`, update `domain-values`, and record it in this file.
  - Result 8.2 (recorded 2026-10-27, GPU build of the Nix-built binary, Qwen2.5-1.5B-Instruct Q4_K_M):
    (a) `djinn-agent --once "What is 12*7? Use the calculator tool."` called `calculator({"expression":"12 * 7"})` and answered 84.
    (b) Ollama `llama3.2:latest` and `qwen2.5:latest` both made valid `calculator` calls (and `llama3.2` a `read_file` call); no tool call was rejected by the strict validation, so no rule had to be relaxed (D6).
    (c) Every 400 scenario returned HTTP 400 with the value kind, the JSON path and the rule, e.g. ``messages[1].tool_calls[0].function.arguments: tool arguments `[1,2]`: must be a JSON object``. `/complete` bad parameters return 422 (the status the server already uses for an unparsable body), naming the parameter.
    Found and fixed on the way: `POST /complete {"prompt":""}` was a 500 on the real CUDA model (zero-token input; `CUDA_ERROR_INVALID_VALUE`), although the `inference-server-api` spec says an empty prompt is accepted. It only passed in unit tests with a mock model. An empty prompt now yields an empty completion (`200 {"prompt":"","output":""}`). Not caused by this change, but covered by it. Also: the stop-limit message read `stop: stop: ...`; the kind is now `stop sequences`.
- [x] 8.3 Because `Cargo.lock` changes (`url`), verify `nix build .#djinn-cli` still succeeds (long build; run last).
- [x] 8.4 `openspec validate --all --strict` passes.

## 9. Instructions and documentation

- [x] 9.1 `CLAUDE.md`: the "Conventions" section (already drafted with this plan) is accurate for the finished code — correct the type list and module paths if they drifted, then **delete the sentence saying the migration is in progress**.
- [x] 9.2 `README.md`: add a short note on validated settings (what an invalid `--ip`/`--base-url`/config value looks like) and the new JSON Schema constraints; verify the documented commands run as written.
- [x] 9.3 `openspec/config.yaml`: confirm the `context:` entry (already added with this plan) still matches; adjust if a decision changed during implementation.

## Workflow follow-up

- Owner reviews with `jj diff`, commits each group, then runs `openspec archive typed-domain-values` (this syncs the delta specs into `openspec/specs/`, creating `domain-values`).
- Not done here and worth a later change: typing `repeat_penalty` / `repeat_last_n`, and a `clippy::disallowed_methods` rule restricting `from_static` to tests and constants.
