# djinn-cli-commands Specification

## Purpose
Lists the subcommands of the `djinn-cli` binary so that scripts and docs know
what is supported, and records that the image-similarity command is gone.

## Requirements

### Requirement: Supported subcommands
`djinn-cli` SHALL provide the subcommands `server` (run the HTTP server from
command-line arguments), `server-config` (run the HTTP server from a named config
file), `single-run` (run one local generation, optionally saving its settings as a
config), and `config` (run one local generation from a saved config). It SHALL exit non-zero with a usage message for an unknown subcommand.

#### Scenario: Help lists the commands
- **WHEN** a user runs `djinn-cli --help`
- **THEN** the output lists `server`, `server-config`, `single-run`, and `config`

#### Scenario: Server from a named config
- **WHEN** a user runs `djinn-cli server-config --name test`
- **THEN** the server starts using `configs/server/test.toml`

### Requirement: Clip subcommand removed
`djinn-cli` SHALL NOT provide a `clip` subcommand.

#### Scenario: Clip is rejected
- **WHEN** a user runs `djinn-cli clip`
- **THEN** the command exits non-zero and reports an unrecognised subcommand

### Requirement: Hardware features are selected on djinn-models
Building `djinn-cli` with GPU acceleration SHALL use the `djinn-models` feature
flags (`djinn-models/mac`, `djinn-models/cuda`), and the project's documented
build and run recipes SHALL use those names.

#### Scenario: Documented recipe builds
- **WHEN** the `server` recipe in the `justfile` is run on Apple Silicon
- **THEN** it passes `--features djinn-models/mac` and the build succeeds

### Requirement: Command-line values are validated when parsed
`djinn-cli` SHALL validate each argument that names an address, a file, a
config, or a Hub identifier while parsing the command line (see
`domain-values`). An invalid value SHALL make the command exit non-zero with a
usage error that names the argument, before any server starts or model loads.

#### Scenario: Bad listen address
- **WHEN** a user runs `djinn-cli server --ip not-an-ip`
- **THEN** the command exits non-zero with a usage error naming `--ip`, and no socket is opened

#### Scenario: Config name escapes the config directory
- **WHEN** a user runs `djinn-cli server --model-config ../elsewhere`
- **THEN** the command exits non-zero naming `--model-config`

#### Scenario: Config name with extension
- **WHEN** a user runs `djinn-cli server-config --name test.toml`
- **THEN** the command exits non-zero and says to give the name without `.toml`

#### Scenario: Defaults and documented examples still work
- **WHEN** the commands in the README and the `justfile` are run unchanged
- **THEN** their arguments are accepted
