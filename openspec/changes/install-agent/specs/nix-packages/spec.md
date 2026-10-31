## ADDED Requirements

### Requirement: The flake exports the agent
The flake SHALL export a `djinn-agent` package that provides `bin/djinn-agent` with the terminal UI enabled, and SHALL declare it as the package's main program so that `nix run .#djinn-agent` starts it. The package SHALL NOT need a GPU, CUDA or any hardware feature.

#### Scenario: Run without installing
- **WHEN** a user runs `nix run .#djinn-agent -- --version`
- **THEN** the command prints the `djinn-agent` version and exits 0

#### Scenario: Install on a machine
- **WHEN** the package is added to `environment.systemPackages` or a user profile
- **THEN** `djinn-agent` is on `PATH` and `djinn-agent --help` lists the flags `--base-url`, `--model`, `--plain`, `--approval` and `--tool-root`

#### Scenario: Defaults reach a local djinn server
- **WHEN** an installed `djinn-agent` is started with no flags and no `DJINN_AGENT_*` variables
- **THEN** it targets `http://localhost:8080/v1`, the default address of the `services.djinn` module

### Requirement: Server package variants
The flake SHALL export the server in explicit hardware variants on Linux: `djinn-cli-cpu` (no hardware feature, no CUDA toolchain) and `djinn-cli-cuda` (CUDA). The existing `djinn-cli` and `default` outputs SHALL keep their meaning: CUDA on Linux, Metal on macOS.

#### Scenario: CPU build has no CUDA dependency
- **WHEN** `djinn-cli-cpu` is built
- **THEN** its binary does not link any CUDA library

#### Scenario: Default package unchanged
- **WHEN** `nix build .#default` is run on Linux
- **THEN** it builds the CUDA-enabled `djinn-cli`

### Requirement: Packages are checked to start
`nix flake check` SHALL build the `djinn-agent` package and run `djinn-agent --version` and `djinn-agent --help` from the built output, failing if either exits non-zero.

#### Scenario: A package that does not start fails the check
- **WHEN** the built `djinn-agent` binary exits non-zero for `--version`
- **THEN** `nix flake check` fails and names the smoke check

### Requirement: Installation is documented
The README SHALL explain how to install the agent (flake input and package, or `nix profile install`), how it finds the server (flags, `DJINN_AGENT_*` variables and the default), and that `read_file` is confined to the working directory unless `--tool-root` is given.

#### Scenario: The documented install works
- **WHEN** a user follows the README's install steps on a NixOS machine running `services.djinn`
- **THEN** `djinn-agent` starts and talks to the local server with no further configuration
