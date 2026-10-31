# Tasks

Verification commands run inside `nix develop --command ...` (Rust) or from the repository root (`nix ...`; use `path:.` while new files are untracked). Do not stage or commit; the owner commits with `jj`.

## 1. Package

- [x] 1.1 `flake.nix`: add the `djinn-agent` package per design D1 and export it in `packages` for every system (D2), with `meta.mainProgram`. Verify with `nix build path:.#djinn-agent` and `result/bin/djinn-agent --version`. The agent's tests (insta snapshots included) run inside the build, so the package source set also keeps `*.snap` files (crane's filter drops them).
- [x] 1.2 Verify the build reuses the CPU dependency artifacts: `nix build path:.#djinn-agent` after `djinn-cli-cpu` is built compiles no candle, tokenizers or hf-hub crate (the build log shows only the agent's own dependency set and the two workspace crates it needs), and `nix run path:.#djinn-agent -- --help` lists `--base-url`, `--model`, `--plain`, `--approval`, `--tool-root`.

## 2. Check

- [x] 2.1 `flake.nix` (`checks`): add `djinn-agent-starts` per design D3. Verify `nix flake check path:.` passes, and that changing a grepped flag name in the check makes it fail.

## 3. Docs

- [x] 3.1 `README.md`: an install section under `djinn-agent` (flake input, `environment.systemPackages` snippet, `nix profile install`, `nix run`), the server defaults and `DJINN_AGENT_*` variables, and the `--tool-root` note (design D5). `CLAUDE.md`: one line listing the flake's package outputs. Verify the snippet evaluates by adding it to a throwaway `nixosSystem` evaluation (`nix eval` of `config.environment.systemPackages` containing `djinn-agent`), and `just check` stays green.
- [x] 3.2 `openspec validate --all --strict` passes.

## Workflow follow-up

- The owner pushes, then adds the package to the machine configuration (dotfiles) and updates the flake lock: `environment.systemPackages = [ inputs.djinn.packages.${pkgs.system}.djinn-agent ];`.
- Live check: `djinn-agent` on a machine running `services.djinn` answers a message with no flags; record it at the end of this file.
- Archive the change after review.
