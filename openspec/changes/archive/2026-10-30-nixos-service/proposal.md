## Why

The djinn server can only be started by hand (`djinn-cli server ...` from a checkout). The owner runs NixOS machines and wants the server to be a managed systemd service, configured the way every other NixOS service is: `services.djinn = { enable = true; port = 8080; ... };`.

## What Changes

- The flake exports `nixosModules.default` (also `nixosModules.djinn`), a NixOS module that defines `services.djinn` and, when enabled, a `djinn.service` systemd unit running `djinn-cli server`.
- Options: `enable`, `package`, `address`, `port`, `openFirewall`, `logLevel`, `extraEnvironment`, `environmentFile`, `cuda.enable` (off by default) and a fully declarative `model` submodule (architecture, Hugging Face repo/revision/weights file/tokenizer repo, or local files). The module renders the model options to a TOML file in the Nix store and points the server at it. No hand-written config file is needed.
- CUDA is opt-in. The module defaults to a CPU build; `cuda.enable = true` selects the CUDA build, sets `device = "cuda"` in the model config and gives the unit access to the GPU devices. Nothing in the module assumes an NVIDIA machine.
- The flake adds a CPU-only package (`djinn-cli-cpu`) and an explicit CUDA package (`djinn-cli-cuda`). The existing `djinn-cli`/`default` outputs keep their current meaning (CUDA on Linux, Metal on macOS).
- The unit is hardened: a dynamic user, a persistent state directory that holds the Hugging Face cache (`HF_HOME`), and the usual sandboxing options.
- `djinn-cli server` honours `--config-dir`. Today the model config path is always `./configs/<name>.toml` relative to the working directory and `--config-dir` is ignored for it, which a service (no checkout, no meaningful working directory) cannot use. **BREAKING** only for a caller who passed `--config-dir` to `server` and relied on it being ignored.
- Docs: a "NixOS" section in the README with a complete example.

## Capabilities

### New Capabilities

- `nixos-module`: the `services.djinn` NixOS module: options, the systemd unit it produces, the generated model config, GPU opt-in, hardening and state directory.

### Modified Capabilities

- `djinn-cli-commands`: `server` resolves the model config under `--config-dir` (added requirement; the existing requirements are unchanged).

## Impact

- `flake.nix`: new `nixosModules`, new `djinn-cli-cpu` / `djinn-cli-cuda` packages, a `checks` entry that evaluates the module.
- New `nix/module.nix` (the module itself, so `flake.nix` stays readable).
- `djinn-cli/src/server.rs`: build the model config path from `--config-dir` (+ tests).
- `README.md`, `CLAUDE.md` (one line on the module), `openspec/specs/`.
- No change to the HTTP API, the config file formats or the JSON schemas. No new Rust dependencies.
- Out of scope: a VM test that downloads real weights (needs network), TLS or authentication in front of the server, running the agent as a service, Darwin (launchd) support, and non-Hugging-Face model sources beyond the existing `Files` source.
