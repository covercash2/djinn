# CLAUDE.md

## Commands

```sh
just check          # typos + core-deps guard + fmt + clippy + tests (run before committing)
just schema         # regenerate configs/**/*.schema.json (run after changing any config struct)
just agent          # run the agent REPL against a running server (see README)
just coverage       # HTML coverage report → target/llvm-cov/html/index.html
cargo nextest run -p <crate> <test_name>  # single test
```

Dev environment: Nix flakes (`.envrc` runs `use flake`). Rust nightly required.

## Workspace

`djinn-core`, `djinn-models`, `djinn-server`, `djinn-agent`, `djinn-cli`, `xtask` are in the workspace. `ollama-cli` is standalone (separate `Cargo.toml`/`Cargo.lock`, not a workspace member).

- `djinn-core` — shared, dependency-light code: config loading/validation, the serializable config types (`RunConfig`, `ModelConfig`, `Device`, ...), and the OpenAI-compatible chat wire types (`djinn_core::chat`) shared by the server and the agent. **Must not depend on candle, tokenizers, or hf-hub**; `just check` enforces this (`just check-core-deps`).
- `djinn-models` — everything that links candle: model loading, token streaming, the `LanguageModel` trait, and the hardware features.
- `djinn-server` — axum HTTP API over a loaded `LanguageModel`: raw completion (`/complete`) and OpenAI-compatible chat with tool calls (`/v1/chat/completions`).
- `djinn-agent` — the agent harness (REPL, tool loop, `Backend` and `Tool` traits). Talks to any OpenAI-compatible server over HTTP. **Must not depend on candle, `djinn-models`, or `djinn-server`**; `just check` enforces this (`just check-agent-deps`).
- `djinn-cli` — the `djinn-cli` binary (`server`, `server-config`, `single-run`, `config`).

Work is planned and tracked as OpenSpec changes under `openspec/`; the owner commits (with `jj`), so do not stage or commit on their behalf.

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
