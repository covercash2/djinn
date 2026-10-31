# Design

## Context

See proposal.md. Observed state:

- `flake.nix` builds with crane. `commonArgs` is the Linux CUDA / macOS Metal build; `cpuArgs` (added by `nixos-service`) is the same without any hardware feature or CUDA inputs, with its own `cpuArtifacts = craneLib.buildDepsOnly cpuArgs`. Packages today: `djinn-cli`, `djinn-cli-cpu`, `djinn-cli-cuda` (Linux), `ollama-cli`, `default`.
- `djinn-agent` is a workspace member with one binary and a default-on `tui` feature. It depends on neither candle nor `djinn-models` nor `djinn-server` (enforced by `just check-agent-deps`), so it needs none of the CUDA/Metal inputs.
- Its CLI already reads `DJINN_AGENT_BASE_URL`, `DJINN_AGENT_MODEL` and `DJINN_AGENT_API_KEY`; the default base URL is `http://localhost:8080/v1`, the `services.djinn` default (agent-repl spec, "Configuration").

## Goals / Non-Goals

**Goals**
- `nix run .#djinn-agent` and `environment.systemPackages = [ djinn.packages.${system}.djinn-agent ]` work.
- Adding the package does not rebuild the candle stack.
- A broken package fails `nix flake check`, not the first use.

**Non-Goals**
- Changing the agent (flags, defaults, config files), or installing it from the NixOS module.
- Home Manager module, completions, man pages, macOS verification.

## Decisions

### D1. One package, built from the existing CPU dependency artifacts

`djinn-agent = craneLib.buildPackage (cpuArgs // { cargoArtifacts = cpuArtifacts; pname = "djinn-agent"; cargoExtraArgs = "-p djinn-agent"; meta.mainProgram = "djinn-agent"; })`. `cpuArtifacts` is the CPU build of the whole workspace, so the candle stack is not compiled again. Cargo still rebuilds the agent's own, small dependency set (about 150 crates: reqwest, tokio, ratatui and so on) because feature unification differs when only `-p djinn-agent` is built and because the tests are built too; that is minutes, not the candle build. No separate feature variants: the agent has no hardware features.

*Alternative:* `commonArgs`/`cargoArtifacts` (the CUDA set). Rejected: it drags the CUDA toolchain into a package that does not use it and would make the agent unbuildable where CUDA is not available.

### D2. Defined for every system, verified on x86_64-linux

`cpuArgs` has no platform-specific inputs, so the package is defined in the common part of `packages`, not under `optionalAttrs isLinux`. On macOS the C toolchain from the default stdenv supplies the SDK; this has not been built and is recorded as unverified in the proposal.

### D3. Smoke check as a derivation that runs the built binary

`checks.<system>.djinn-agent-starts = pkgs.runCommand "djinn-agent-starts" { } ''${djinn-agent}/bin/djinn-agent --version; ${djinn-agent}/bin/djinn-agent --help | grep -q -- --base-url; touch $out''`. It builds the package as a side effect, so `nix flake check` covers "builds" and "starts". `--help` is grepped for the flags the spec promises (`--base-url`, `--tool-root`), which also catches an accidental flag rename. The terminal UI is not started: that needs a tty and is covered by the existing Rust tests.

*Alternative:* `nixosTest` running the agent against a stub server. Rejected as disproportionate: the agent's behaviour is already tested in Rust; this change only has to prove the packaging.

### D4. Spec the flake's package set in one capability

`nixos-service` added `djinn-cli-cpu`/`djinn-cli-cuda` in `flake.nix` but only described them in its design. They are part of what the flake promises (the NixOS module's default package depends on them), so `nix-packages` records them alongside `djinn-agent`.

### D5. Installation docs say what is surprising

An installed binary behaves differently from `just agent` in one way that matters: `--tool-root` defaults to `.`, so `read_file` can read whatever is under the directory the user starts the agent in (and nothing outside it). The README states this next to the install steps, and shows the dotfiles-style snippet for NixOS.

## Risks / Trade-offs

- [`cpuArtifacts` is shared with the CPU server build] A change to workspace dependencies rebuilds both. → Intended; they are one dependency graph.
- [Smoke check only proves `--version`/`--help`] A package that starts but fails on a terminal is not caught. → The UI is covered by `TestBackend` and scripted-session tests in Rust.
- [macOS unverified] → Stated as out of scope.

## Migration Plan

Additive. After pushing, the owner adds `djinn.packages.${system}.djinn-agent` to the machine configuration and updates the flake lock. Rollback is removing it.
