# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```sh
# Run full check suite (typos, clippy, tests)
just check

# Build and run the server on Apple Silicon (recommended for Mac)
cargo run --release --features djinn-core/mac -- server-config --name test

# Run the ollama TUI
just tui

# Run tests for all workspace members
cargo test --all

# Run a single test
cargo test -p <crate-name> <test_name>

# Lint
cargo clippy

# Check for typos
typos
```

The project uses Nix flakes for dev environment setup (`.envrc` runs `use flake`).

## Architecture Overview

This is a Cargo workspace with these members:

- **`djinn-core`** — Core ML library. Contains model loading, inference logic, device abstraction, and image processing. This is the heart of the project.
- **`djinn-cli`** — Main binary entrypoint (`djinn`). Parses CLI args and dispatches to server or single-run inference.
- **`djinn-server`** — Axum HTTP server. Endpoints: `GET /health-check`, `POST /complete`, `POST /clip`. Swagger UI at `/swagger-ui`; OpenAPI JSON at `/api-doc/openapi.json`. Serves an HTMX frontend from `./djinn-server/assets/`.
- **`ollama-cli`** — Standalone Ratatui TUI for interacting with an Ollama server. Has its own separate `Cargo.toml`/`Cargo.lock` and is not part of the main workspace.

### Key Data Flow

1. CLI parses args → selects `Runner` subcommand
2. `server-config` subcommand reads `./configs/server/<name>.toml` → loads `Config` (socket addr + model config path)
3. `run_server` reads the model config TOML → calls `create_new_context` → builds `ModelContext` (tokenizer + weights + device)
4. HTTP requests hit `/complete` → acquire `Mutex<Context>` → `ModelContext::run()` returns a token stream → collects to string → JSON response

### `djinn-core` Module Layout

- `lm/` — Language model abstractions
  - `model.rs` — `ModelArchitecture` enum (Mistral, QMistral, Starcoder), `Model` enum, `ModelContext` (runs inference via `async_stream`)
  - `config.rs` — `ModelConfig` (variant, device, flash_attn, model_source), `RunConfig` (sampling params), `ModelRun`
  - `mistral/` — Mistral-specific HF Hub loading and context creation
- `image/` — Image utilities, vision encoders, and Stable Diffusion pipeline
  - `mod.rs` — `save_image`, `load_image`, `load_images` (tensor ↔ file); `VisionEncoder` trait; shared `VisionEncoderError` / `VisionEncoderResult` used by all encoders
  - `clip.rs` — CLIP encoder (`Clip`, `ClipArgs`); `ModelFile` (also holds `siglip_tokenizer`/`siglip_model` constructors)
  - `siglip.rs` — SigLIP encoder (`SigLip`, `SigLipArgs`)
  - `vision_encoder_ext.rs` — Free functions over `&dyn VisionEncoder`: `text_image_similarity`, `rank_texts_by_image`
  - `gen.rs` — Stable Diffusion image generation (v1.5, v2.1, SDXL, Turbo, inpainting variants); `Args` struct is the CLI entrypoint
  - `config.rs` — `GenConfig` (serde TOML struct); loaded from `~/.config/djinn/image-gen.toml`
  - `error.rs` — `Error` / `Result<T>` for the image gen pipeline (typed `thiserror` variants)
- `config/` — Generic XDG config infrastructure
  - `mod.rs` — `Error` / `Result<T>` for config loading
  - `xdg.rs` — `config_dir()`, `config_file()`, `load<T>()` — reads TOML from `$XDG_CONFIG_HOME/djinn/`
- `device.rs` — `Device` enum (Cpu/Cuda/Metal) with `cfg_if` feature-gated defaults
- `hf_hub_ext.rs` — HuggingFace Hub helpers (`Hub` wrapper, `hub_load_safetensors`)
- `yolov8/` — YOLOv8 object detection
- `error.rs` — Top-level `Error` type for the crate

### Feature Flags (`djinn-core`)

- `mac` — Enables Metal GPU and Accelerate framework for Apple Silicon
- `cuda` — Enables CUDA GPU acceleration
- `fixed-seed` — Uses a fixed seed (`299792458`) instead of a random one
- `openapi` — Derives `utoipa::ToSchema` on core types (e.g. `RunConfig`); enabled automatically by `djinn-server`

### Config Files

Server configs live in `./configs/server/<name>.toml` and reference model configs at paths like `./configs/model/<model>.toml`.

#### Schema Maintenance

All config structs derive `schemars::JsonSchema`. Runtime validation (via `djinn_core::config::validate_and_load`) is always in sync with the Rust type — no manual step needed. The `configs/**/*.schema.json` files are for editor/Taplo tooling only and must be regenerated whenever a config struct changes:

```sh
just schema
```

Run this after adding, removing, or renaming fields on any of:
- `djinn_server::Config` → `configs/server/server.schema.json`
- `djinn_core::lm::config::ModelConfig` → `configs/model/model.schema.json`
- `djinn_core::lm::config::RunConfig` → `configs/lm/run-config.schema.json`
- `djinn_core::lm::config::ModelRun` → `configs/lm/model-run.schema.json`
- `djinn_core::image::config::GenConfig` → `configs/image-gen.schema.json`

When adding a **new** config type, also: derive `JsonSchema` on the type, add it to `djinn-cli/src/bin/schema-gen.rs`, and add a `#:schema <relative-path>` header to the TOML template.

Image generation config lives at `~/.config/djinn/image-gen.toml` (XDG). All fields are optional `Option<T>`. Priority order: config file < env var (`DJINN_SD_*`) < explicit CLI flag.

Example model config (`configs/model/q_mistral.toml`):
```toml
variant = "q_mistral"
flash_attn = false

[model_source.hugging_face_hub]
revision = "main"
```

### Toolchain

Rust **nightly** is required (see `rust-toolchain.toml`). The `djinn-cli` main uses `#![feature(addr_parse_ascii)]`.

### Logging

`RUST_LOG` controls log levels. Default filter: `warn,djinn_server=debug,djinn_core=debug,axum=debug`. Pass `--tracing chrome` to emit a `trace-<timestamp>.json` for Chrome's `chrome://tracing`.

### `ollama-cli` Notes

This crate is standalone (not in the workspace) and has a local path dependency on `modelfile` at `/Users/chrash/github/covercash2/modelfile/`. It uses `ratatui`/`crossterm` for the TUI, `ollama-rs` for the Ollama API client, and `insta` for snapshot tests.
