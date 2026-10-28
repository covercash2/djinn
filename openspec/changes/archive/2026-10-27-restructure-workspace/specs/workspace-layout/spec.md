# Spec Delta

## Purpose

Defines the crate boundaries of the djinn workspace so that shared code stays
light, inference-runtime code is isolated in one place, and unsupported model
families do not accumulate. Other crates and later changes rely on these rules.

## ADDED Requirements

### Requirement: djinn-core has no inference-runtime dependencies
`djinn-core` SHALL NOT depend, directly or transitively through its own
dependencies list, on candle, tokenizers, or hf-hub. It SHALL contain only
config loading and validation, config-surface types, and its own error type.

#### Scenario: Dependency tree is free of the inference stack
- **WHEN** `cargo tree -p djinn-core` is run with default features
- **THEN** the output contains none of `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`, `hf-hub`

#### Scenario: Core builds without hardware features
- **WHEN** `cargo build -p djinn-core` is run on a machine with no GPU toolchain
- **THEN** the build succeeds

### Requirement: djinn-models owns model execution
A workspace crate named `djinn-models` SHALL own model loading, token generation,
the language-model abstraction used by the server, and the conversion from the
config-surface `Device` to a runtime device. The `mac` and `cuda` feature flags
SHALL be defined on `djinn-models` and not on `djinn-core`.

#### Scenario: Hardware feature selects the compute backend
- **WHEN** `djinn-models` is built with `--features mac` (or `cuda`)
- **THEN** the corresponding candle acceleration is enabled for the whole build

#### Scenario: Old feature path is gone
- **WHEN** a build is requested with `--features djinn-core/mac`
- **THEN** cargo reports that `djinn-core` has no such feature

### Requirement: Only text-generation models are supported
The set of supported model variants SHALL contain only language models that the
server can run for text generation. Object detection, vision encoders, image
generation, code-specialised models, and non-generative encoder models SHALL NOT
be present in the workspace.

#### Scenario: Variant list is text-only
- **WHEN** the model-variant enumeration is inspected (`configs/model/model.schema.json`)
- **THEN** every listed variant is a text-generation language model, and none is a detection, vision, image-generation, code-specialised, or encoder-only model

#### Scenario: Vision code is removed
- **WHEN** the workspace source is searched for `yolov8`, `siglip`, `clip`, or image-generation modules
- **THEN** none are found outside `openspec/` and version-control history

### Requirement: Config schemas stay in sync
Every config type that derives `JsonSchema` SHALL have a generated schema file
under `configs/`, and no schema file SHALL exist for a type that was removed.
Running the schema task SHALL leave the working tree unchanged.

#### Scenario: Schema task is idempotent
- **WHEN** `just schema` is run on a clean checkout
- **THEN** `git diff --exit-code configs/` (or `jj diff configs/`) reports no changes

#### Scenario: Removed type leaves no schema
- **WHEN** the files under `configs/` are listed
- **THEN** there is no `image-gen.toml` or `image-gen.schema.json`

### Requirement: Workspace verification passes
The project's verification command SHALL pass after the restructure with the
same or stricter strictness as before: typos, formatting, clippy with warnings
denied across all workspace targets, and the full test suite.

#### Scenario: just check is green
- **WHEN** `just check` is run inside the Nix dev shell
- **THEN** it exits 0

#### Scenario: Coverage threshold still enforced
- **WHEN** the coverage step of `just check` runs
- **THEN** it enforces the existing minimum line coverage (30%) and passes
