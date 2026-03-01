# CLAUDE.md

## Commands

```sh
just check          # typos + clippy + tests (run before committing)
just schema         # regenerate configs/**/*.schema.json (run after changing any config struct)
just coverage       # HTML coverage report → target/llvm-cov/html/index.html
cargo nextest run -p <crate> <test_name>  # single test
```

Dev environment: Nix flakes (`.envrc` runs `use flake`). Rust nightly required.

## Workspace

`djinn-core`, `djinn-server`, `djinn-cli`, `xtask` are in the workspace. `ollama-cli` is standalone (separate `Cargo.toml`/`Cargo.lock`, not a workspace member).

## Config & Schema

- Server configs: `./configs/server/<name>.toml`; model configs: `./configs/model/<model>.toml`
- All config structs derive `schemars::JsonSchema`. `just schema` regenerates editor schema files via `xtask`.
- Run `just schema` after adding, removing, or renaming fields on any config struct.
- When adding a new config type: derive `JsonSchema`, add to `xtask/src/main.rs`, add `#:schema <path>` header to TOML template.

## Feature Flags (`djinn-core`)

- `mac` — Metal + Accelerate (Apple Silicon)
- `cuda` — CUDA
- `fixed-seed` — deterministic seed `299792458`
- `openapi` — derives `utoipa::ToSchema` on core types; auto-enabled by `djinn-server`

## Logging

Default `RUST_LOG`: `warn,djinn_server=debug,djinn_core=debug,axum=debug`. Pass `--tracing chrome` to emit `trace-<timestamp>.json` for `chrome://tracing`.
