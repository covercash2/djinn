## ADDED Requirements

### Requirement: The flake exports a NixOS module
The flake SHALL export `nixosModules.default` (and the same module as `nixosModules.djinn`). Importing it SHALL define the options under `services.djinn` and SHALL NOT start or install anything until `services.djinn.enable = true`.

#### Scenario: Imported but disabled
- **WHEN** a NixOS configuration imports the module and leaves `services.djinn.enable` unset
- **THEN** no `djinn` systemd service, user, firewall rule or state directory is defined

#### Scenario: Enabled with defaults
- **WHEN** a configuration sets only `services.djinn.enable = true`
- **THEN** it evaluates, and `systemd.services.djinn` is defined, wanted by `multi-user.target`

### Requirement: Service options
The module SHALL provide these options under `services.djinn`: `enable`, `package`, `address`, `port`, `openFirewall`, `logLevel`, `extraEnvironment`, `environmentFile`, `cuda.enable` and `model`. `port` SHALL be a TCP port (1–65535), `address` an IP address, and an out-of-range port or a malformed address SHALL fail at evaluation time, not at service start. The defaults SHALL be: address `127.0.0.1`, port `8080`, `openFirewall` false, `cuda.enable` false.

#### Scenario: Port and address reach the server
- **WHEN** `port = 9000` and `address = "0.0.0.0"`
- **THEN** the unit's command line is `djinn-cli server --ip 0.0.0.0 --port 9000 ...`

#### Scenario: Invalid port rejected
- **WHEN** `port = 70000`
- **THEN** evaluating the configuration fails with a message naming `services.djinn.port`

#### Scenario: Firewall is opt-in
- **WHEN** `openFirewall` is false (the default)
- **THEN** `networking.firewall.allowedTCPPorts` does not contain the service port
- **AND WHEN** `openFirewall = true` **THEN** it contains the port

### Requirement: Declarative model configuration
The module SHALL describe the model with options and render them to a TOML model config that the server reads, so no hand-written config file is required. The rendered file SHALL deserialize as the server's model config without changes to that format.

#### Scenario: Default model
- **WHEN** the service is enabled and `model` is left at its defaults
- **THEN** the rendered config selects `qwen2_5_instruct` from the Hugging Face Hub at revision `main`

#### Scenario: Overridden source
- **WHEN** `model.source.huggingFace.repo = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"` and `model.source.huggingFace.weightsFile = "qwen2.5-0.5b-instruct-q4_k_m.gguf"`
- **THEN** the rendered config contains that repo and weights file

#### Scenario: Rendered config is accepted by the server
- **WHEN** the rendered config of a representative configuration is loaded by `djinn-server`'s config validation
- **THEN** it loads without error

### Requirement: Model sources
The `model` options SHALL cover the architecture (default `qwen2_5_instruct`) and a source that is either the Hugging Face Hub (revision, optionally repo, weights file, tokenizer repo and tokenizer revision) or local files (weight files and a tokenizer file). Setting both sources SHALL fail at evaluation time.

#### Scenario: Local files
- **WHEN** `model.source.files` gives weight files and a tokenizer file
- **THEN** the rendered config uses the files source and has no Hugging Face section

#### Scenario: Both sources set
- **WHEN** `model.source.files` is set and a Hugging Face option is changed from its default
- **THEN** evaluation fails with a message naming `services.djinn.model.source`

### Requirement: CUDA is opt-in
The module SHALL NOT enable GPU acceleration by default. With `cuda.enable = false` the service SHALL use a CPU build and no GPU device access. With `cuda.enable = true` the service SHALL use the CUDA build, SHALL set `device = "Cuda"` in the rendered model config, and SHALL be allowed to open the NVIDIA and render/video device nodes. An explicit `package` SHALL take precedence over the package the module would choose.

#### Scenario: Default is CPU
- **WHEN** the service is enabled without `cuda.enable`
- **THEN** the unit runs the CPU package, the rendered config has no `device` entry, and the unit has no GPU device allowances

#### Scenario: CUDA requested
- **WHEN** `cuda.enable = true`
- **THEN** the unit runs the CUDA package, the rendered config has `device = "Cuda"`, and the unit's device allowances include the NVIDIA device nodes

#### Scenario: Explicit package wins
- **WHEN** `package` is set and `cuda.enable = true`
- **THEN** the unit runs the given package

### Requirement: Service runtime and persistence
The unit SHALL run `djinn-cli server` as an unprivileged dynamic user with a persistent state directory (`/var/lib/djinn`) holding the Hugging Face cache (via `HF_HOME`), so weights are downloaded once. It SHALL wait for the network and restart on failure.

#### Scenario: Weights cache persists
- **WHEN** the service restarts
- **THEN** it reuses the cache under `/var/lib/djinn` and does not download the weights again

### Requirement: Service environment
The unit SHALL set `RUST_LOG` from `logLevel`, merge `extraEnvironment`, and read `environmentFile` (for secrets such as a Hugging Face token) without placing its contents in the Nix store. `extraEnvironment` SHALL NOT override `HF_HOME` or `RUST_LOG`.

#### Scenario: Reserved variables
- **WHEN** `extraEnvironment` sets `HF_HOME`
- **THEN** evaluation fails with a message naming `services.djinn.extraEnvironment`

#### Scenario: Secrets stay out of the store
- **WHEN** `environmentFile = "/run/secrets/djinn.env"` is set
- **THEN** the unit has `EnvironmentFile=/run/secrets/djinn.env` and no secret value appears in the generated unit or config

#### Scenario: Restart on failure
- **WHEN** the server process exits with an error
- **THEN** systemd restarts it after a delay

### Requirement: Service hardening
The unit SHALL apply systemd sandboxing: no new privileges, a private `/tmp`, a read-only system and home, protected kernel settings, and a restricted set of address families. Device access SHALL be limited to what `cuda.enable` grants.

#### Scenario: Hardened by default
- **WHEN** the module is enabled
- **THEN** the unit sets `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem = "strict"` and `ProtectHome`, and has no `DeviceAllow` entries unless `cuda.enable` is true
