## Why

`djinn-agent` is the front end for the djinn server, and the server now runs as a NixOS service (`nixos-service`). But the agent can only be started from a checkout (`just agent`, `cargo run -p djinn-agent`): the flake exports no package for it, so it cannot be installed on a machine, put on `PATH`, or run with `nix run`.

## What Changes

- The flake exports a `djinn-agent` package (the `djinn-agent` binary, terminal UI included) and `nix run .#djinn-agent` works. It has no hardware features, so one build serves every machine.
- The flake check builds the package and runs `djinn-agent --version` and `--help` from the built output, so a package that builds but does not start is caught.
- The package build reuses the CPU build of the dependencies, so adding it does not compile the candle stack again (only the agent's own small dependency set).
- The README says how to install it (flake input and `environment.systemPackages` / `nix profile install`) and what a user must know about an installed binary: the defaults already match a local djinn server (`http://localhost:8080/v1`), `DJINN_AGENT_*` variables set the rest, and `read_file` is confined to the directory the agent is started in (`--tool-root`).
- The package outputs that the NixOS module already relies on (`djinn-cli-cpu`, `djinn-cli-cuda`) were added by `nixos-service` without a spec; this change records them in the new capability so the flake's package set is specified in one place.
- No Rust code changes, no change to the CLI, the HTTP API, or any config format.

## Capabilities

### New Capabilities

- `nix-packages`: which packages the flake exports (`djinn-cli`, `djinn-cli-cpu`, `djinn-cli-cuda`, `djinn-agent`, `ollama-cli`), what each contains, and the checks that prove they build and start.

### Modified Capabilities

None. The existing `agent-repl` configuration requirement already specifies the flags, variables and defaults the installed binary uses.

## Impact

- `flake.nix`: new `djinn-agent` package and a smoke check.
- `README.md` (install section), `CLAUDE.md` (one line), `openspec/specs/`.
- Out of scope: installing the agent from the NixOS module (`services.djinn` stays server-only), a Home Manager module, per-user configuration files, shell completions, and verifying the macOS build (the package is defined for every system but only checked on x86_64-linux). Adding the package to the owner's machine configuration happens in the dotfiles repository, after this change is pushed.
