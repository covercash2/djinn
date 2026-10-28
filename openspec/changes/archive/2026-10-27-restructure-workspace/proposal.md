# Proposal

> **Sequence:** this is change **1 of 2**. It must land before
> `add-agent-harness`, which builds the agent harness on the layout produced
> here. This change is a *behavior-preserving refactor plus removals*: no new
> features. After it, `just check` is green and the text-generation server
> behaves exactly as it does today.

## Why

`djinn-core` is supposed to hold what the other crates share, but it has become
the home of every model implementation (Mistral, StarCoder, YOLOv8, CLIP,
SigLIP, Stable-Diffusion-style image generation) and depends on candle,
tokenizers, hf-hub, `image`, `imageproc` and `ab_glyph`. Every crate that wants a
config type or an error enum pays for the entire inference stack.

We are about to build an agent harness (`djinn-agent`, change 2) that must stay
independent of the inference runtime and talk to *any* backend. Most of the
existing models (object detection, vision encoders, image generation, a code
model, a base non-chat language model) do not serve a chat-and-tools workflow
and are dead weight there. Clearing them out first keeps change 2 small and
makes the crate boundaries something we can enforce.

## What Changes

- **New crate `djinn-models`** owns everything that links candle: model
  loading, the `LanguageModel` trait, token streaming, the HF Hub helpers, the
  `Device -> candle Device` conversion, and the `mac` / `cuda` feature flags.
- **`djinn-core` becomes candle-free.** It keeps only shared, dependency-light
  code: config loading/validation (`config`, `config::xdg`), the config-surface
  types (`RunConfig`, `ModelConfig`, `ModelRun`, `ModelSource`,
  `ModelArchitecture`, `Device` as a plain enum), and its own `Error`. It keeps
  the `fixed-seed` and `openapi` features.
- **Remove models and code that do not serve the agent workflow:**
  - `yolov8` (object detection), `coco_classes`, `font`
  - `image` module: CLIP, SigLIP, `VisionEncoder`, image generation (`gen`),
    and `image::config` (`GenConfig`)
  - language-model variants `Mistral` (full-precision, base), `Starcoder`,
    `DistilBert` from `ModelArchitecture`
  - **kept for now:** `QMistral` (quantized base Mistral 7B). It is the only
    text model the server can run today; it is removed by change 2 once an
    instruct model replaces it.
- **BREAKING** `POST /clip` route and the CLIP model load at server start-up are
  removed. The server no longer loads CLIP; start-up is faster and needs less
  memory.
- **BREAKING** `djinn-cli clip` subcommand removed.
- **BREAKING** `configs/image-gen.toml`, `configs/image-gen.schema.json`, and the
  `GenConfig` schema entry in `xtask` are removed.
- **BREAKING** the `mac` / `cuda` feature flags move from `djinn-core` to
  `djinn-models`. Build invocations change from `--features djinn-core/mac` to
  `--features djinn-models/mac` (justfile, flake, CLAUDE.md, README).
- Split `djinn_core::Error`: the candle and tokenizer variants move to a
  `djinn_models::Error`; core keeps config/parameter errors.

Anything removed here is recoverable from version control history. Nothing is
being *rewritten*; removed code is deleted, moved code is moved with its tests.

## Capabilities

### New Capabilities

- `workspace-layout`: the crate boundaries and dependency rules that every
  other change relies on: what `djinn-core` may and may not depend on, what
  `djinn-models` owns, where feature flags live, and which model families are
  supported.
- `inference-server-api`: the HTTP surface of `djinn-server` as of this change
  (health check, raw completion, SSE streaming completion, OpenAPI docs, 404
  fallback) and, explicitly, the absence of `/clip` and of any static frontend.
  Change 2 adds the chat route as a separate capability rather than editing
  this one.
- `djinn-cli-commands`: the subcommands `djinn-cli` exposes (server,
  server-config, single-run, config) and the removal of `clip`.

### Modified Capabilities

<!-- None: openspec/specs/ is empty. All three capabilities above are introduced
     here, describing current behavior plus the removals. -->

## Impact

- **Crates:** new `djinn-models`; `djinn-core`, `djinn-server`, `djinn-cli`,
  `xtask` change imports and `Cargo.toml` dependencies. `ollama-cli` is
  untouched (standalone, not a workspace member).
- **Dependencies dropped from the workspace:** `image`, `imageproc`, `ab_glyph`,
  `base64` (only used by `/clip`), and the candle/tokenizers/hf-hub edges from
  `djinn-core`. Verify each with `cargo udeps` after the move (the dev shell
  provides it).
- **Files/dirs removed:** `djinn-core/data/font/`, `djinn-core/src/{yolov8,image}/`
  and their snapshots, `djinn-server/src/clip/`, `djinn-cli/src/clip.rs`,
  `configs/image-gen.*`.
- **Build/CI/docs touched:** `justfile` (`server` recipe feature flag),
  `flake.nix` (`cargoExtraArgs` feature flags), `.github/workflows/rust.yml`
  (clippy scope), `CLAUDE.md` (workspace + feature-flag sections), `README.md`,
  `typos.toml` if needed.
- **Generated artifacts:** `just schema` must be re-run; schema files must
  match (CI already checks `git diff --exit-code configs/`).
- **Risk:** low-medium. Mostly mechanical moves; the real risks are visibility
  (`pub(crate)` items that now cross a crate boundary) and the feature-flag
  rename silently breaking the Nix build. Both are covered in `design.md`.
