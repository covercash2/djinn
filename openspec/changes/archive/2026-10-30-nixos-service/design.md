# Design

## Context

See proposal.md for motivation. Observed state:

- The flake (`flake.nix`) builds `djinn-cli` with crane. On Linux it always passes `--features cuda` and adds `autoAddDriverRunpath`, so the binary finds `libcuda` in `/run/opengl-driver/lib` at run time. There are no `nixosModules` and no CPU-only package.
- `djinn-cli server` takes `--ip`, `--port`, `--config-dir`, `--model-config` and `--save-config`. `TryFrom<ServerArgs> for Config` (`djinn-cli/src/server.rs`) builds the model path as `./configs/<model_config>.toml` and ignores `config_dir`.
- The model config (`djinn_core::lm::config::ModelConfig`) is `variant`, optional `device` (`Cpu`/`Cuda`/`Metal`, no serde rename, so `"Cpu"`/`"Cuda"`/`"Metal"`), and `model_source` (`hugging_face_hub { revision, repo_id?, weights_file?, tokenizer_repo?, tokenizer_revision? }` or `files { weight_files, tokenizer_file }`). `device` omitted means "best device compiled in".
- Weights come from `hf-hub` at start-up and are cached in the Hugging Face cache (`HF_HOME`, else `~/.cache/huggingface`).

## Goals / Non-Goals

**Goals**
- `services.djinn = { enable = true; port = 9000; };` works with no other files.
- The model is configured in Nix; nothing outside the Nix configuration is needed (except optional secrets).
- GPU use is a deliberate choice per machine.
- The module is testable without a GPU or network.

**Non-Goals**
- A VM test that downloads weights. A live check is a task for the owner's machine.
- TLS, authentication, reverse proxy or a second instance of the service.
- launchd (macOS) support; Home Manager module.
- Changing the model-config file format or the HTTP API.

## Decisions

### D1. Module lives in `nix/module.nix`, exported from the flake as `nixosModules`

`flake.nix` is already long. The module is a function `self: { config, lib, pkgs, ... }:` so it can default `package` from `self.packages.${pkgs.stdenv.hostPlatform.system}`. The flake exports it outside `eachDefaultSystem` (modules are system-independent): `nixosModules.default = import ./nix/module.nix self;`, with `nixosModules.djinn` as an alias.

*Alternative:* put the module inline in `flake.nix`. Rejected: harder to read and to evaluate in a check.

### D2. Packages: add `djinn-cli-cpu` and `djinn-cli-cuda`, keep `djinn-cli`

Today's `djinn-cli`/`default` stay exactly as they are (CUDA on Linux, Metal on macOS) so `nix build`, the dev shell (`inputsFrom`) and existing habits do not change. Two new outputs make the choice explicit:

- `djinn-cli-cpu`: no hardware feature, and no CUDA toolchain in `buildInputs` (a small, quick build; no `autoAddDriverRunpath`).
- `djinn-cli-cuda`: the current Linux build (`--features cuda`).

The module's `package` default is `djinn-cli-cpu`, or `djinn-cli-cuda` when `cuda.enable`. Setting `package` overrides both. On Darwin the module is not applicable, so only the Linux outputs are defined as module defaults; the flake still evaluates on all systems because the CPU/CUDA packages are defined for Linux only (`optionalAttrs isLinux`).

*Alternative:* make `djinn-cli` CPU-only on Linux. Rejected here: it silently changes the owner's existing `nix build`/dev-shell behaviour. Easy to do later once the new names exist.

### D3. The model config is generated from options and the server gets `--config-dir`

Options map one-to-one to `ModelConfig`:

```
services.djinn.model = {
  architecture = "qwen2_5_instruct";           # enum, one value today
  source.huggingFace = {                         # attrs submodule, used by default
    revision = "main";
    repo = null; weightsFile = null;             # null = architecture default
    tokenizerRepo = null; tokenizerRevision = null;
  };
  source.files = null;                           # { weightFiles = [ path ]; tokenizerFile = path; }
};
```

Exactly one of `source.huggingFace` / `source.files` is in effect: `files` set (non-null) wins and an assertion rejects setting both. `pkgs.formats.toml` renders the file (`nix/module.nix` builds an attrset, keys in snake_case, nulls dropped). The file is written to a store directory `…/configs/model/djinn.toml` and the unit runs

```
djinn-cli server --ip <address> --port <port> --config-dir <store dir> --model-config model/djinn
```

Consequences: `--config-dir` must work (D4), the model config name `model/djinn` is the existing `ConfigName` grammar, and nothing about the TOML format changes. `device = "Cuda"` is written only when `cuda.enable` is true; otherwise `device` is omitted and the CPU build picks CPU.

*Alternatives:* (a) a `--model-config-file <path>` flag: a second way to do the same thing and a new CLI surface; (b) `settings` as a free-form attrset (`pkgs.formats.toml`'s usual pattern): flexible, but wrong values fail at service start, not at evaluation. Typed options catch typos where they are written. `extraConfig`-style escape hatches are not provided; add one when a need appears.

### D4. Fix `--config-dir` in `djinn-cli`

`TryFrom<ServerArgs> for Config` joins `config_dir` and `<model_config>.toml` instead of the literal `./configs/`. The default (`./configs`) keeps today's behaviour, so only a caller who passed a custom `--config-dir` and relied on it being ignored changes. The `Config` conversion already has unit tests (`default_args_make_a_valid_config`); add tests for a custom directory and for the default. A missing file already fails in `run_server` (`read_to_string(model_path)?`); wrap it so the error names the path (task 1.2). Types are unchanged: `config_dir` is `PathBuf`, `model_config` is `ConfigName`.

### D5. Option types and validation at evaluation time

| Option | Type | Rule |
|---|---|---|
| `port` | `types.port` | 1–65535 |
| `address` | custom `ipAddress` type (IPv4 dotted quad or something containing `:` that matches an IPv6 pattern) | rejected otherwise; the server's `IpAddr` parse remains the authority at run time |
| `openFirewall`, `cuda.enable`, `enable` | `bool` | |
| `logLevel` | `str`, default `warn,djinn_server=info,djinn_core=info` | passed as `RUST_LOG`; `tracing` filters are free text, not validated |
| `extraEnvironment` | `attrsOf str` | merged over the module's own variables, may not override `HF_HOME` or `RUST_LOG` (assertion) |
| `environmentFile` | `nullOr path` stored as a *string*, not a store path | `types.path` would copy it into the store and leak secrets; use `types.nullOr types.str` with an absolute-path assertion |
| `model.source.huggingFace.*` | `str` / `nullOr str` | repo `owner/name` and file name patterns mirror `RepoId`, `WeightsFile`, `Revision` rules; the server re-validates on load |

Because the Rust newtypes are the real authority (domain-values spec), the module only repeats the cheap shape checks that give early, readable errors, and the golden-file test (D7) proves the output deserializes through the real types.

### D6. systemd unit

```
after/wants = network-online.target; wantedBy = multi-user.target
DynamicUser = true; StateDirectory = "djinn"; WorkingDirectory = "/var/lib/djinn"
Environment: HF_HOME=/var/lib/djinn/huggingface, RUST_LOG, + extraEnvironment
EnvironmentFile = environmentFile (optional)
Restart = "on-failure"; RestartSec = 5
NoNewPrivileges, PrivateTmp, ProtectSystem=strict, ProtectHome, ProtectKernel{Tunables,Modules,Logs},
ProtectControlGroups, RestrictAddressFamilies=[AF_INET AF_INET6 AF_UNIX], RestrictNamespaces,
LockPersonality, RestrictRealtime, SystemCallArchitectures=native
```

First start downloads about 1 GB, so `TimeoutStartSec` is raised (`infinity` is not used; 30 min) and the process is not `Type=notify` (the server does not signal readiness; the unit is `Type=simple`).

With `cuda.enable`: `DeviceAllow` for `/dev/nvidiactl`, `/dev/nvidia*`, `/dev/nvidia-uvm*`, `char-nvidia*`, `/dev/dri/*` (rw), `SupplementaryGroups = [ "video" "render" ]`, `PrivateDevices = false`, and `LD_LIBRARY_PATH` is not needed because `djinn-cli-cuda` already carries the `/run/opengl-driver/lib` RUNPATH. Without it `PrivateDevices = true`. The module asserts `hardware.graphics.enable` (or the legacy `hardware.opengl.enable`) is set when `cuda.enable` is true, so the host driver path exists; the message tells the user what to enable. It does **not** enable the NVIDIA driver itself: that is the machine's decision.

### D7. How the module is verified without a GPU or network

1. `checks.<linux>.nixos-module`: evaluates `nixpkgs.lib.nixosSystem` with the module in four configurations (minimal; custom port/address/firewall; `cuda.enable`; explicit `package`), and asserts the properties in the spec (`ExecStart` args, firewall ports, `DeviceAllow` empty/non-empty, chosen package, environment, `EnvironmentFile`). Failing assertions fail `nix flake check`. Evaluation only — nothing is built or started.
2. A negative check: `port = 70000` and `address = "nope"` fail evaluation (`builtins.tryEval` on the option value).
3. Golden file: `nix/tests/model-default.toml` is the expected rendering for the default model. The same check compares the rendered file with it, and a Rust test in `djinn-core` (`config` tests) parses that file with `validate_and_load::<ModelConfig>`. Format drift on either side breaks one of the two.
4. Live check (owner, optional, recorded in the notes at the end of tasks.md): `nixos-rebuild`/`nixos-container` with the module, `systemctl status djinn`, `curl http://127.0.0.1:<port>/v1/models` (or the health route the server has), then again with `cuda.enable = true` on an NVIDIA machine.

## Risks / Trade-offs

- [DynamicUser plus a persistent HF cache] The state directory is owned by a transient uid that systemd maps; this is the standard pattern and works, but the cache path moves if the user later switches to a static user. → Documented in the README; keep `HF_HOME` under `StateDirectory`.
- [CUDA sandboxing is easy to get slightly wrong] A hardened unit without the right device nodes gives a confusing CUDA init error. → The module's device list is taken from nixpkgs' own CUDA-using services; the live check on an NVIDIA machine is an explicit task, and the failure mode (falls back? errors?) is recorded.
- [`device = "Cuda"` on a CPU-only `package`] Setting `package` to a CPU build with `cuda.enable` makes the server error at start. → An assertion cannot see inside a package, so this is documented on the option; the server's error is clear.
- [Two packages double the CI/build surface] `djinn-cli-cuda` is the current build; `djinn-cli-cpu` is a fresh crane build with its own `buildDepsOnly`. → Acceptable; the CPU build is much smaller. CI is not extended in this change.
- [`--config-dir` fix] Behaviour change for custom `--config-dir`, see D4. → Default unchanged; called out in the proposal as **BREAKING** only in that narrow case.
- [Renamed/new `Device` serde] The `device` string is `"Cuda"` because the enum has no rename. If the enum is ever renamed to snake_case, the module and the golden file change together; the golden test catches it.

## Migration Plan

Additive. A machine adopts the module by adding the flake input and `imports = [ djinn.nixosModules.default ]`. Rollback is removing `services.djinn.enable`. The `--config-dir` fix needs no migration.

## Open Questions

- Which route to use for the live health check (`/v1/models` may not exist yet); settle when running the check, no effect on the module.
