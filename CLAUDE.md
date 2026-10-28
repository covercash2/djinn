# CLAUDE.md

## Commands

```sh
just check          # typos + core-deps guard + fmt + clippy + tests (run before committing)
just schema         # regenerate configs/**/*.schema.json (run after changing any config struct)
just agent          # run the agent REPL against a running server (see README)
just coverage       # HTML coverage report → target/llvm-cov/html/index.html
cargo nextest run -p <crate> <test_name>  # single test
```

Dev environment: Nix flakes (`.envrc` runs `use flake`). Stable Rust, edition 2024; the toolchain version is pinned in `rust-toolchain.toml`, `flake.nix` and `.github/workflows/rust.yml` (bump all three together).

## Workspace

`djinn-core`, `djinn-models`, `djinn-server`, `djinn-agent`, `djinn-cli`, `xtask` are in the workspace. `ollama-cli` is standalone (separate `Cargo.toml`/`Cargo.lock`, not a workspace member).

- `djinn-core` — shared, dependency-light code: config loading/validation, the serializable config types (`RunConfig`, `ModelConfig`, `Device`, ...), and the OpenAI-compatible chat wire types (`djinn_core::chat`) shared by the server and the agent, and the validated value types (`djinn_core::values`). **Must not depend on candle, tokenizers, or hf-hub**; `just check` enforces this (`just check-core-deps`).
- `djinn-models` — everything that links candle: model loading, token streaming, the `LanguageModel` trait, and the hardware features.
- `djinn-server` — axum HTTP API over a loaded `LanguageModel`: raw completion (`/complete`) and OpenAI-compatible chat with tool calls (`/v1/chat/completions`).
- `djinn-agent` — the agent harness (REPL, tool loop, `Backend` and `Tool` traits). Talks to any OpenAI-compatible server over HTTP. **Must not depend on candle, `djinn-models`, or `djinn-server`**; `just check` enforces this (`just check-agent-deps`).
- `djinn-cli` — the `djinn-cli` binary (`server`, `server-config`, `single-run`, `config`).

Work is planned and tracked as OpenSpec changes under `openspec/`; the owner commits (with `jj`), so do not stage or commit on their behalf.

## Conventions

**Use validated newtypes for domain values, not bare `String`/`f64`/`usize`.** Identifiers, names, paths, addresses, URLs, secrets, and bounded parameters each get their own type that is checked on construction **and** on `Deserialize`, so invalid input is rejected where it enters the program: an HTTP body, a TOML config, a CLI argument, or another server's response. Serialized form must not change, and the type's constraints must show up in the JSON Schema/OpenAPI.

- The shared types live in `djinn_core::values` (the module docs have the table of rules): `ToolName`, `ToolCallId`, `ToolArguments`, `ModelId`, `RepoId`, `Revision`, `WeightsFile`, `ConfigName`, `Prompt`, `SystemPrompt`, `StopSequence(s)`, `Temperature`, `TopP`, `MaxTokens`. Agent-only ones are in `djinn-agent/src/settings.rs` (`BaseUrl`, `ApiKey`).
- Prefer a standard type when one exists (`IpAddr`, `PathBuf`, `url::Url`, `StatusCode`). Otherwise add a type with the `string_value!` / `number_value!` macros in `djinn-core/src/values/`: write the validation function, list the type in the module docs, add it to `djinn-core/tests/values_schema.rs` (schema snapshots), and run `just schema`.
- No infallible `From<&str>`/`From<String>` on a type whose conversion can fail. `T::from_static("literal")` (panics) is for literals and tests only.
- Secrets get a redacted `Debug`, no `Display`, and never appear in logs or errors. clap echoes a rejected value in its error, so a secret argument needs a custom `TypedValueParser` (see `ApiKeyParser`).
- `String` stays correct for genuinely free text: message content, tool output, generated text, error messages, the ignored `model` field of a chat request, and values inside a tool's arguments (each tool validates its own).
- Do not re-check a typed value downstream; if you need a rule, put it in the type. `validate_request` only holds rules that no single value can check (non-empty conversation, tool results answering an earlier call).
- When a server must report a value error, the serde message already names the kind (`tool name`, `temperature`, ...); map data errors to 400 (see `ChatError::from(JsonRejection)`).

The rules for each type are in the `domain-values` OpenSpec spec.

## Config & Schema

- Server configs: `./configs/server/<name>.toml`; model configs: `./configs/model/<model>.toml`
- All config structs derive `schemars::JsonSchema`. `just schema` regenerates editor schema files via `xtask`.
- Run `just schema` after adding, removing, or renaming fields on any config struct.
- When adding a new config type: derive `JsonSchema`, add to `xtask/src/main.rs`, add `#:schema <path>` header to TOML template.

## Feature Flags

`djinn-models`:
- `mac` — Metal + Accelerate (Apple Silicon)
- `cuda` — CUDA

`djinn-cli` forwards both (`--features mac` / `--features cuda`); from the workspace root `--features djinn-models/mac` also works.

`djinn-core`:
- `fixed-seed` — deterministic seed `299792458`
- `openapi` — derives `utoipa::ToSchema` on core types; auto-enabled by `djinn-server`
- `clap` — additional `clap::Args` derives on config structs

## Logging

Default `RUST_LOG`: `warn,djinn_server=debug,djinn_core=debug,axum=debug`. Pass `--tracing chrome` to emit `trace-<timestamp>.json` for `chrome://tracing`.
