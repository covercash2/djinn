# Design

## Context

Read `proposal.md` for motivation and `specs/` for the behavior contracts. This
file is the *how*.

Observed state of the workspace (verified against the source, not assumed):

- `djinn-core` is ~4.4k lines. Roughly 2.4k of those are vision/detection code
  (`yolov8/` 1.2k, `image/gen.rs` 0.9k, CLIP/SigLIP ~0.6k) that only the
  `/clip` route and `djinn-cli clip` use.
- The language-model path is small: `lm/{mod,config,model}.rs`,
  `lm/mistral/mod.rs`, `token_output_stream.rs`, `hf_hub_ext.rs`, `device.rs`.
- Config-surface types live next to runtime code. `ModelConfig` (in
  `lm/config.rs`) holds a `Device` and `ModelArchitecture`; `ModelRun` holds a
  `ModelConfig` and `RunConfig`. These are serde + `JsonSchema` types and are
  pure data. They do **not** need candle.
- `djinn_core::Error` has `#[from] candle_core::Error` and
  `#[from] tokenizers::Error` variants, which is what drags candle into every
  user of the error type.
- `djinn-server` already depends on `djinn-core` with `features = ["openapi"]`
  and uses it for `LanguageModel`, `RunConfig`, `ModelConfig`, `VisionEncoder`,
  `validate_and_load`, and `create_new_context`.
- `xtask` generates five schemas; one (`GenConfig`) goes away.
- Hardware flags `mac` / `cuda` are defined on `djinn-core`; the `justfile`
  (`--features djinn-core/mac`) and `flake.nix` (`--features mac` / `cuda`)
  reference them.

Tooling note for implementers: `cargo` is only on the PATH inside the Nix dev
shell. Run commands as `nix develop --command <cmd>` (or via direnv). The
flake's toolchain pin was updated in a prior session so that the lockfile
compiles; do not downgrade it.

## Goals / Non-Goals

**Goals:**

- `cargo tree -p djinn-core` contains no candle, tokenizers, or hf-hub.
- All model execution lives in one new crate, `djinn-models`.
- Zero behavior change for `/complete`, `/complete/stream`, `/health-check`,
  `/swagger-ui`, and `djinn-cli {server,server-config,single-run,config}`,
  apart from the explicit removals.
- `just check` is green after **each** task group, not only at the end.

**Non-Goals:**

- No new models, no chat API, no agent code (that is change 2,
  `add-agent-harness`).
- No renaming of public config fields or TOML keys. Existing
  `configs/model/q_mistral.toml` and `configs/server/test.toml` must keep
  loading unchanged.
- No upgrade of candle (stays 0.8.4) and no change to the pinned toolchain.
- No changes to `ollama-cli`.

## Decisions

### D1. New `djinn-models` crate rather than moving models into `djinn-server`

Chosen: a new workspace crate `djinn-models`.

Why: `djinn-cli single-run` and `config` run models *without* the HTTP stack;
change 2 adds chat templating and tool-call parsing that belongs next to the
model, not next to axum; and the `mac`/`cuda` features and the heavy candle
build stay in one crate. It also lets `djinn-server` shrink to routing and
state. A later `djinn-agent` integration test can depend on `djinn-models`
without the server.

Alternative considered: fold models into `djinn-server`. Rejected: it forces the
CLI to link the whole HTTP server just to run one prompt, and makes the server
crate carry two responsibilities. It is the smaller diff, but the dependency
direction (CLI → server → candle) is worse.

### D2. What goes where

| Item (current path in `djinn-core/src/`) | Destination |
|---|---|
| `config/mod.rs`, `config/xdg.rs` (`validate_and_load`, `DEFAULT_CONFIG_DIR`, XDG helpers) | stays in `djinn-core` |
| `lm/config.rs` (`RunConfig`, `ModelConfig`, `ModelRun`, defaults, `load_config`) + its snapshot test | stays in `djinn-core` |
| `device.rs`: the `Device` enum, its `Default`, `ValueEnum`, `JsonSchema` | stays in `djinn-core` |
| `device.rs`: conversion to a candle device | moves to `djinn-models` as a free function (see D3) |
| `lm/mod.rs`: `ModelSource` enum | stays in `djinn-core` |
| `lm/mod.rs`: `LanguageModel` trait | `djinn-models` |
| `lm/model.rs`: `ModelArchitecture` enum + `hf_repo_id()` | enum + `hf_repo_id()` stay in `djinn-core` (pure data); `hf_files()`, `Model`, `ModelContext`, builder, generation loop move to `djinn-models` |
| `lm/mistral/mod.rs` (`create_new_context`, `run`, `run_model`) | `djinn-models` |
| `token_output_stream.rs` | `djinn-models` |
| `hf_hub_ext.rs` | `djinn-models`, minus `hub_load_safetensors` if nothing uses it after pruning (it only served full-precision Mistral and StarCoder) |
| `error.rs` | split: `djinn-core::Error` keeps config/parameter variants; candle/tokenizer/anyhow variants go to `djinn_models::Error` |
| `yolov8/`, `coco_classes.rs`, `font.rs`, `data/font/`, `image/`, `tensor_ext.rs` | **deleted** (`tensor_ext` is only used by the CLIP paths) |
| `lib.rs` `#![feature(coverage_attribute)]` + `extern crate accelerate_src` under `mac` | the `extern crate` moves to `djinn-models`; `coverage_attribute` is kept wherever `#[coverage(off)]` is used |

Module paths inside `djinn-models` should mirror what exists (`lm`, `device`,
`token_output_stream`) so diffs stay reviewable.

### D3. Device conversion cannot be a `TryFrom` impl after the split

Today `impl TryFrom<Device> for candle_core::Device` lives in `djinn-core`
where `Device` is local. Once `Device` stays in core and the conversion moves
to `djinn-models`, **both types are foreign to `djinn-models`**, so Rust's orphan
rule forbids the impl. Use a plain function
`djinn_models::device::to_candle(Device) -> candle_core::Result<candle_core::Device>`
(or an extension trait) and update the two call sites
(`lm/mistral/mod.rs`, `djinn-server/src/lib.rs`'s former CLIP init disappears).

### D4. Keep `QMistral`; prune the rest of `ModelArchitecture`

Remove variants `Mistral`, `Starcoder`, `DistilBert`. Keep `QMistral` with its
serde name `q_mistral` unchanged so existing TOML keeps parsing. Prune
`Model` (the runtime enum in `lm/model.rs`) to the single remaining variant;
keep it an enum so change 2 can add a variant without reshaping the code.
Delete `configs/model/starcoder.toml`. `djinn-core`'s `hf_repo_id` unit test
shrinks accordingly.

Why keep QMistral at all: the server must keep starting and passing tests with a
real model path between the two changes. Change 2 removes it.

### D5. Error types

`djinn_models::Error` = `Candle`, `Tokenizer`, `Anyhow`, `ParameterFileParse`
(whichever of today's variants the remaining code actually produces).
`djinn_core::Error` shrinks to what core code actually returns. Core's current
`Error` is only used for non-config errors, so check with the compiler: after
removing candle the unused variants will surface as dead-code or compile errors.
`djinn-server::Error::Core(djinn_core::Error)` becomes
`Error::Models(djinn_models::Error)` (plus keep a core variant only if the
server still uses core errors, e.g. for config loading). Remove
`Error::Base64` and `Error::VisionEncoder` and their tests along with `/clip`.
The 500-with-generic-message behavior in `error.rs` must be preserved
(spec: "Model failure").

### D6. Server start-up and state after removing CLIP

`AppState` becomes `{ model }` only. `build_router` loses `fast_router`'s clip
route; `/health-check` keeps the 1-second timeout layer. `run_server` stops
constructing `Clip`. `openapi.rs` drops the clip schemas/paths. Remove the
`base64` dependency (verify with `cargo udeps`). The `test_state` / `test_app`
helpers in `server/mod.rs` tests lose the mock `VisionEncoder`.

### D7. Feature flags and build recipes

- `djinn-models` defines `mac` (metal + accelerate-src + candle `metal`
  features) and `cuda` exactly as `djinn-core` does today.
- Add forwarding features to `djinn-cli`: `mac = ["djinn-models/mac"]`,
  `cuda = ["djinn-models/cuda"]`. This is needed because `flake.nix` currently
  passes `--features mac|cuda` with `-p djinn-cli`, which requires the feature to
  exist on that package. **Before changing anything**, run `nix build .#djinn-cli
  --dry-run` and, if time allows, a real `nix build` to record whether the flake
  already works on this machine; do not claim it still works unless it was
  built.
- `justfile` `server` recipe: `--features djinn-models/mac` (works through the
  dependency) or the forwarding `--features mac`; pick one and document it in
  `CLAUDE.md` and `README.md`.
- `fixed-seed` stays on `djinn-core` (it controls `RunConfig`'s default seed);
  `openapi` stays on `djinn-core`. Update the "Feature Flags" section of
  `CLAUDE.md` to say which crate owns each flag.

### D8. Order of work keeps the tree green

Do deletions first (task group 2), then the split (groups 3-4), then
build/docs (group 5). Deletions are the lowest-risk way to shrink the surface
and each leaves the workspace compiling. Do not do the crate split with the
vision code still present; it doubles the moves.

## Risks / Trade-offs

- **Visibility leaks across the new boundary** (`pub(crate)` items in
  `lm/model.rs` used by `lm/mistral`) → both move together into
  `djinn-models`, so nothing public needs to widen. If the compiler asks for
  `pub`, prefer `pub(crate)` inside `djinn-models`.
- **Orphan-rule surprise** for the device conversion → addressed in D3.
- **`#![feature(coverage_attribute)]` and `#[coverage(off)]`** are nightly
  features used on network/GPU code. The new crate needs the crate-level
  `#![feature(coverage_attribute)]` or `just check`'s clippy/coverage step fails
  to compile. → Copy the attribute into `djinn-models/src/lib.rs`.
- **Nix build drift**: renaming the feature path can break `nix build`
  silently because it is not part of `just check`. → D7 baseline-then-verify.
- **Coverage threshold**: removing well-tested vision code (image config tests
  are ~100% covered) shifts the ratio. Current total is ~73% vs a 30% gate, so
  there is ample headroom; still re-run `just check` and report the number.
- **Schema drift**: removing `GenConfig` and the enum variants changes
  `model.schema.json`. → Re-run `just schema` and commit the regenerated files
  in the same group (CI diffs `configs/`).
- **False positives from `cargo udeps`** (only default features are checked).
  → For each dependency removed, also grep for `cfg(feature = ...)` use first.
- **Unrecoverable removals**: deletions are final in the working tree. They are
  recoverable from VCS history; the user commits themselves, so do not run
  git/jj write commands (see Tasks 1.1).

## Migration Plan

Single-step, no data migration. Users of the removed pieces:

1. `POST /clip` callers: no replacement. Document the removal in `README.md`.
2. `djinn-cli clip` users: none expected; no replacement.
3. Anyone building with `--features djinn-core/mac|cuda`: switch to
   `--features djinn-models/mac|cuda` (or the forwarding `mac` / `cuda` on
   `djinn-cli`).

Rollback: revert the working-copy changes via the VCS; no persistent state is
changed.
