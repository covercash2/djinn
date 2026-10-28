## ADDED Requirements

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
