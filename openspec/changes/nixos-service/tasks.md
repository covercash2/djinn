# Tasks

Verification commands run inside `nix develop --command ...`; `nix` checks run from the repository root. Do not stage or commit; the owner commits with `jj`.

## 1. Server honours `--config-dir`

- [x] 1.1 `djinn-cli/src/server.rs`: build the model config path as `config_dir.join("<model_config>.toml")` (design D4). Tests: a custom `--config-dir` gives `<dir>/model/custom.toml`; the default still gives `./configs/model/qwen2_5_instruct.toml`. Verify with `cargo nextest run -p djinn-cli`.
- [x] 1.2 `djinn-server/src/lib.rs`: when reading the model config fails, the error names the resolved path (add context, e.g. `anyhow::Context`). Verify with a test that starts from a missing path, or by a `tests/cli.rs`-style check if the function stays excluded from coverage; `just check` stays green.
- [x] 1.3 Golden model config: add `nix/tests/model-default.toml` (default Qwen model, Hugging Face source) and a `djinn-core` test that loads it with `validate_and_load::<ModelConfig>` and asserts the architecture and source. Verify the test passes and that a deliberately wrong key makes it fail.

## 2. Packages

- [x] 2.1 `flake.nix`: add `djinn-cli-cpu` (no hardware feature, no CUDA inputs) and `djinn-cli-cuda` (today's Linux build) for Linux; leave `djinn-cli`, `ollama-cli` and `default` unchanged (design D2). Verify with `nix build .#djinn-cli-cpu` and `nix build .#djinn-cli-cuda`, then run `result/bin/djinn-cli --help` for each; `nix build .#djinn-cli` still produces the same derivation path as before the change, unless inputs differ.

## 3. The module

- [x] 3.1 Create `nix/module.nix` with the option set from design D5: `enable`, `package`, `address`, `port`, `openFirewall`, `logLevel`, `extraEnvironment`, `environmentFile`, `cuda.enable`, `model`; defaults as in the spec. Verify `nix eval` of a minimal configuration (task 4.1) succeeds and `port = 70000` / `address = "nope"` fail.
- [x] 3.2 Render `model` to TOML into a store directory `configs/model/djinn.toml` with `pkgs.formats.toml` (design D3), dropping nulls and writing `device = "Cuda"` only with `cuda.enable`. Verify the rendering equals `nix/tests/model-default.toml` for the default configuration.
- [x] 3.3 Define `systemd.services.djinn`, the firewall rule and the assertions (design D3, D5, D6): command line, environment, state directory, hardening, CUDA device access, `hardware.graphics` assertion, `model.source` exclusivity. Verify with the evaluation checks in 4.1.
- [x] 3.4 `flake.nix`: export `nixosModules.default` and `nixosModules.djinn` (design D1). Verify `nix flake show` lists them and `nix flake check` evaluates.

## 4. Module checks

- [x] 4.1 `flake.nix` `checks.<linux>.nixos-module` (implemented in `nix/tests/module-check.nix`): evaluate four configurations (minimal; custom port, address and firewall; `cuda.enable`; explicit `package` with `cuda.enable`) and assert every scenario in `specs/nixos-module/spec.md`, including the golden-file comparison and the two negative cases (`builtins.tryEval`). Verify with `nix flake check`, and that flipping one expected value makes the check fail.
- [x] 4.2 `just check` (Rust side) is green, and `openspec validate --all --strict` passes.

## 5. Docs and live check

- [x] 5.1 `README.md`: a "NixOS" section: add the flake input, import the module, a complete example (`services.djinn = { enable = true; port = 9000; model.source.huggingFace.revision = "main"; };`), the CUDA opt-in with the `hardware.graphics` requirement, how weights are cached, and `environmentFile` for a token. `CLAUDE.md`: one line about `nix/module.nix` and the `nix flake check` verification. Verify the README example evaluates by pasting it into the check configuration (4.1).
- [x] 5.2 Live check on a NixOS machine (the owner's, or `nixos-container`): enable the module, `systemctl status djinn`, request the server, restart and confirm no second download; then `cuda.enable = true` on an NVIDIA machine. Record the observations at the end of this file. Leave committing and archiving to the owner.

## Live verification notes (5.2, partial)

On the owner's NixOS machine (RTX 4090), with the module imported from the dotfiles `hoss` configuration (`services.djinn = { enable = true; port = 8080; cuda.enable = true; }`):

- `nix build` of the whole system toplevel succeeds. The generated `djinn.service` has the expected `ExecStart` (`djinn-cli-cuda`, `--ip 127.0.0.1 --port 8080 --config-dir <store> --model-config model/djinn`), `HF_HOME`, `DynamicUser`, `StateDirectory`, the NVIDIA `DeviceAllow` lines and the `video`/`render` groups. The rendered `model/djinn.toml` is `device = "Cuda"`, `variant = "qwen2_5_instruct"`, hub revision `main`.
- The same `ExecStart` run by hand as the owner's user: the model loads in about 1.5 s, `GET /v1/models` lists `qwen2.5-1.5b-instruct-q4_k_m`, and `POST /v1/chat/completions` answers. The process maps `libcuda`, `libcublas` and `/dev/nvidia*`, so the CUDA build really runs on the GPU. `nvidia-smi` reported a driver/library mismatch at the time (kernel module older than the installed driver); CUDA initialisation still worked.
- Not yet done: running it as the systemd unit (needs `sudo nixos-rebuild switch`), so the sandbox (`DeviceAllow`, `ProtectSystem`, state directory) and the no-second-download restart are unverified.
- Found when the unit first ran: it failed with `Read-only file system (os error 30)`. `hf_hub::api::tokio::Api::new()` ignores `HF_HOME` and writes under `~/.cache`, which the sandbox makes read-only. The loader now uses `ApiBuilder::from_env()`; verified by running the server with `HF_HOME` set and seeing the cache appear there. (The by-hand run above hid this because the owner's `~/.cache` already held the weights.)
- After the `HF_HOME` fix and `nixos-rebuild switch`: `djinn.service` is active under systemd (dynamic user, hardened, `cuda.enable = true`) and `GET /v1/models` returns `qwen2.5-1.5b-instruct-q4_k_m`. Confirmed by the owner. Not separately checked: that a restart skips the download (the journal shows only the earlier failed starts from before the fix).
