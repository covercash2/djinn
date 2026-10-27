# Tasks

Conventions for whoever implements this (human or agent):

- Run every cargo/just command inside the Nix dev shell, e.g.
  `nix develop --command just check`. `cargo` is not on the plain PATH.
- **Do not commit or stage.** The repository owner makes all commits and uses
  `jj`. Use plain `rm`/`mv`/editing for file changes. Inspect with
  `jj st` / `jj diff`, never `git add|rm|mv|commit`.
- `just check` must pass at the end of **every** group below. If it does not,
  fix it before moving on; do not stack broken groups.
- `design.md` has the destination table (D2) and the orphan-rule note (D3).

## 1. Baseline

- [ ] 1.1 Run `nix develop --command just check` on the untouched tree and record the test count (expect 56 passed) and total line coverage (expect ~73%) in a scratch note. Verify: the command exits 0.
- [ ] 1.2 Record whether the Nix build works today: run `nix build .#djinn-cli --no-link` (CUDA download is large; if it cannot be run, say so explicitly instead of guessing). Verify: result (pass/fail/not run) noted for use in task 5.4.
- [ ] 1.3 Run `nix develop --command cargo tree -p djinn-core --depth 1 | grep -E "candle|tokenizers|hf-hub"` and note the lines. Verify: non-empty output now (it must be empty after group 4).

## 2. Remove vision, detection and image generation

- [ ] 2.1 Delete `djinn-server/src/clip/`, the `/clip` route and `ServiceRoutes::Clip`, the `fast_router` clip wiring, the `Clip` construction and `VisionEncoder` state in `run_server`, the clip entries in `djinn-server/src/openapi.rs`, and `Error::Base64` / `Error::VisionEncoder` plus their tests in `djinn-server/src/error.rs`. `AppState` keeps only the model. Verify: `cargo test -p djinn-server` passes and the `clip_*` tests are gone.
- [ ] 2.2 Delete `djinn-cli/src/clip.rs`, the `Runner::Clip` variant and its match arm in `djinn-cli/src/main.rs`. Verify: `cargo run -p djinn-cli -- clip` fails with an unrecognised-subcommand error and `djinn-cli --help` lists only server, server-config, single-run, config.
- [ ] 2.3 Delete `djinn-core/src/{yolov8,image}/`, `coco_classes.rs`, `font.rs`, `tensor_ext.rs`, `djinn-core/data/font/`, the image snapshot file, and their `mod` lines in `lib.rs`. Verify: `cargo check --workspace --all-targets` passes.
- [ ] 2.4 Remove `GenConfig` from `xtask/src/main.rs`, delete `configs/image-gen.toml` and `configs/image-gen.schema.json`, then run `just schema`. Verify: `jj diff --stat configs/` shows only the deleted image-gen files and expected model-schema changes, and a second `just schema` run produces no further diff.
- [ ] 2.5 Prune `ModelArchitecture` to `QMistral` only (remove `Mistral`, `Starcoder`, `DistilBert`), prune the runtime `Model` enum in `lm/model.rs` and its match arms to the single variant, delete `configs/model/starcoder.toml`, drop `hub_load_safetensors` if now unused, update the `hf_repo_id` test, and rerun `just schema`. Verify: `configs/model/model.schema.json` lists only `q_mistral` and `cargo test -p djinn-core` passes.
- [ ] 2.6 Remove dependencies that are now unused (`image`, `imageproc`, `ab_glyph`, `base64`, and any others `cargo udeps` reports) from crate and workspace `Cargo.toml`. For each one removed, grep for `cfg(feature` uses first. Verify: `nix develop --command cargo udeps --workspace --all-targets` reports nothing new for the crates touched, and `just check` exits 0.

## 3. Create `djinn-models`

- [ ] 3.1 Add `djinn-models` to the workspace `members` and `[workspace.dependencies]`; create `djinn-models/Cargo.toml` (version/edition/authors from workspace) with the candle, candle-nn, candle-transformers, tokenizers, hf-hub, futures, async-stream, tokio, tokio-stream, tracing, anyhow, thiserror, serde dependencies it needs, plus `djinn-core`, and the `mac` and `cuda` features copied from `djinn-core` (including the optional `metal` and `accelerate-src` deps). Create `src/lib.rs` with `#![feature(coverage_attribute)]` and `extern crate accelerate_src` under `cfg(feature = "mac")`. Verify: `cargo check -p djinn-models` passes.
- [ ] 3.2 Move `token_output_stream.rs`, `hf_hub_ext.rs`, `lm/mistral/mod.rs`, and the runtime parts of `lm/model.rs` (`Model`, `ModelContext`, builder, generation loop, `hf_files`) and the `LanguageModel` trait into `djinn-models`, keeping module names (`lm`, `token_output_stream`, ...). Their existing unit tests move with them. Verify: `cargo test -p djinn-models` runs the moved tests and they pass.
- [ ] 3.3 Add `djinn_models::device::to_candle(Device)` replacing the `TryFrom<Device> for CandleDevice` impl (see design D3), with a unit test for the `Cpu` case. Verify: the test passes and no `TryFrom` impl for a candle type remains in `djinn-core`.
- [ ] 3.4 Create `djinn_models::Error` with the candle/tokenizer/anyhow variants and use it in the moved code. Verify: `cargo check -p djinn-models` passes with no use of `djinn_core::Error` for candle errors.

## 4. Make `djinn-core` candle-free and rewire dependents

- [ ] 4.1 Remove `candle-*`, `tokenizers`, `hf-hub`, and the `mac`/`cuda` features from `djinn-core/Cargo.toml`; shrink `djinn_core::Error` to the variants core still returns; delete `djinn-core/src/device.rs`'s candle import (keep the plain `Device` enum). Then remove every other dependency that `cargo udeps` now reports as unused (likely `async-stream`, `futures`, `tokio-stream`, `tokio`, `anyhow`); `rand` and `cfg-if` stay if `RunConfig` still uses them. Verify: `cargo tree -p djinn-core | grep -E "candle|tokenizers|hf-hub"` prints nothing and `cargo test -p djinn-core` passes (including the `run-config.toml` snapshot test).
- [ ] 4.2 Point `djinn-server` at `djinn-models` for `LanguageModel`, `create_new_context`, and the error type; add `djinn-models` to its `Cargo.toml`; keep `djinn-core` for `RunConfig`, `ModelConfig`, `validate_and_load` (with `openapi`). Change `Error::Core` to the models error as per design D5. Verify: `cargo test -p djinn-server` passes; the 500-with-generic-message test still passes unchanged in meaning.
- [ ] 4.3 Point `djinn-cli` at `djinn-models` for `run`, `run_model` and the device conversion; add the forwarding `mac` / `cuda` features (`design.md` D7). Verify: `cargo check -p djinn-cli` passes and `cargo run -p djinn-cli -- --help` lists the four commands.
- [ ] 4.4 Point `xtask` at the new locations (config types still come from `djinn-core`/`djinn-server`). Verify: `just schema` runs and produces no diff.
- [ ] 4.5 Add a CI-style guard so the boundary cannot regress: a test or `just` recipe (e.g. in `justfile` `check`, or a small `xtask` subcommand) that fails if `cargo tree -p djinn-core` contains `candle`, `tokenizers`, or `hf-hub`. Verify: temporarily adding `candle-core` to `djinn-core` makes it fail; removing it makes it pass.

## 5. Build, docs, and final verification

- [ ] 5.1 Update `justfile` (`server` recipe feature flag, clippy already `--workspace --all-targets`), `.github/workflows/rust.yml` (use `cargo clippy --workspace --all-targets -- -D warnings`), and `flake.nix` (`cargoExtraArgs`, keep the `-p` crate selection coherent with D7). Verify: `nix develop --command just --dry-run server` shows the new flag and `nix flake check --no-build` evaluates.
- [ ] 5.2 Update `CLAUDE.md`: the Workspace section (add `djinn-models`, add `djinn-agent` as "planned, see openspec"), the Feature Flags section (which crate owns `mac`, `cuda`, `fixed-seed`, `openapi`), and the single-test command. Update `README.md`: crate list, routes table (no `/clip`), and the server run example. Verify: grep the repo for `djinn-core/mac`, `/clip`, and `image-gen` and find no stale references outside `openspec/`.
- [ ] 5.3 Run the full gate: `nix develop --command just check`. Verify: exit 0; report test count and coverage next to the baseline from 1.1 (expect fewer tests because vision tests were removed; coverage must stay above 30%).
- [ ] 5.4 Re-run the Nix build check recorded in 1.2 (`nix build .#djinn-cli --no-link` or `--dry-run` if the baseline could only be a dry run). Verify: result is no worse than the baseline; if the baseline was "not run", state that the Nix build remains unverified rather than claiming it works.
- [ ] 5.5 Smoke test the real server path once if model weights are cached or downloadable: start `just server`, `curl localhost:8080/health-check`, `curl -X POST localhost:8080/complete -H 'content-type: application/json' -d '{"prompt":"hello","sample_len":5}'`, and `curl -X POST localhost:8080/clip` (expect 404). Verify: 200/200/404. If weights cannot be fetched, say so.

## Workflow follow-up

- Archive this change (`openspec archive restructure-workspace`) once the owner has reviewed and committed the work; then start `add-agent-harness`.
