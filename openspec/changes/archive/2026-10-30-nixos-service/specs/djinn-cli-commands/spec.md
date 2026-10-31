## ADDED Requirements

### Requirement: The server honours the config directory
`djinn-cli server` SHALL resolve its model config as `<config-dir>/<model-config>.toml`, where `--config-dir` defaults to `./configs`. It SHALL NOT use a fixed `./configs/` directory when `--config-dir` is given. A model config that does not exist SHALL be reported as an error naming the resolved path.

#### Scenario: Custom config directory
- **WHEN** a user runs `djinn-cli server --config-dir /etc/djinn --model-config model/custom`
- **THEN** the server loads `/etc/djinn/model/custom.toml`

#### Scenario: Default directory unchanged
- **WHEN** a user runs `djinn-cli server` with no `--config-dir`
- **THEN** the server loads `./configs/model/qwen2_5_instruct.toml`

#### Scenario: Missing config is reported with its path
- **WHEN** the resolved model config file does not exist
- **THEN** the command exits non-zero and the error names that path
