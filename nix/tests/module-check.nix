# Evaluation-only check of `nixosModules.default` (openspec nixos-service, specs/nixos-module).
# Nothing is built or started; a failed expectation aborts evaluation with the list of failures.
{ self, nixpkgs, pkgs, system }:

let
  lib = nixpkgs.lib;

  base = {
    nixpkgs.hostPlatform = system;
    boot.loader.grub.enable = false;
    fileSystems."/" = { device = "none"; fsType = "tmpfs"; };
    system.stateVersion = "25.05";
  };

  eval = module: nixpkgs.lib.nixosSystem {
    modules = [ self.nixosModules.default base module ];
  };

  cpu = self.packages.${system}.djinn-cli-cpu;
  cuda = self.packages.${system}.djinn-cli-cuda;

  disabled = eval { };
  minimal = eval { services.djinn.enable = true; };
  custom = eval {
    services.djinn = {
      enable = true;
      port = 9000;
      address = "0.0.0.0";
      openFirewall = true;
      logLevel = "debug";
      extraEnvironment.FOO = "bar";
      environmentFile = "/run/secrets/djinn.env";
      model.source.huggingFace = {
        repo = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
        weightsFile = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
        tokenizerRepo = "Qwen/Qwen2.5-0.5B-Instruct";
      };
    };
  };
  gpu = eval {
    services.djinn = { enable = true; cuda.enable = true; };
    hardware.graphics.enable = true;
  };
  gpuOwnPackage = eval {
    services.djinn = { enable = true; cuda.enable = true; package = cpu; };
    hardware.graphics.enable = true;
  };
  localFiles = eval {
    services.djinn = {
      enable = true;
      model.source.files = {
        weightFiles = [ "/models/q.gguf" ];
        tokenizerFile = "/models/tokenizer.json";
      };
    };
  };
  # Configurations that evaluate but trip an assertion.
  gpuNoDriver = eval { services.djinn = { enable = true; cuda.enable = true; }; };
  bothSources = eval {
    services.djinn = {
      enable = true;
      model.source.files = { weightFiles = [ "/m.gguf" ]; tokenizerFile = "/t.json"; };
      model.source.huggingFace.revision = "v2";
    };
  };
  reservedEnv = eval { services.djinn = { enable = true; extraEnvironment.HF_HOME = "/x"; }; };

  service = s: s.config.systemd.services.djinn;
  exec = s: (service s).serviceConfig.ExecStart;
  sc = s: (service s).serviceConfig;
  noCtx = builtins.unsafeDiscardStringContext;
  has = needle: hay: lib.hasInfix (noCtx needle) (noCtx hay);
  failedMessages = s: map (a: a.message) (lib.filter (a: !a.assertion) s.config.assertions);
  failsWith = s: needle: lib.any (has needle) (failedMessages s);
  # Option values are lazy: force the value to see whether the type check throws.
  rejects = module:
    !(builtins.tryEval (builtins.deepSeq (eval module).config.services.djinn.port true)).success;
  golden = builtins.fromTOML (builtins.readFile ../tests/model-default.toml);

  tests = [
    { name = "disabled defines no service, user or firewall rule";
      ok = !(disabled.config.systemd.services ? djinn)
        && !(lib.elem 8080 disabled.config.networking.firewall.allowedTCPPorts); }
    { name = "enabled with defaults is wanted by multi-user.target";
      ok = (service minimal).wantedBy == [ "multi-user.target" ]; }
    { name = "defaults: 127.0.0.1:8080, firewall closed, cpu package";
      ok = has "--ip 127.0.0.1 --port 8080" (exec minimal)
        && has "${cpu}/bin/djinn-cli" (exec minimal)
        && !(lib.elem 8080 minimal.config.networking.firewall.allowedTCPPorts); }
    { name = "port and address reach the command line";
      ok = has "--ip 0.0.0.0 --port 9000" (exec custom); }
    { name = "openFirewall opens the port";
      ok = lib.elem 9000 custom.config.networking.firewall.allowedTCPPorts; }
    { name = "port 70000 is rejected"; ok = rejects { services.djinn = { enable = true; port = 70000; }; }; }
    { name = "address \"nope\" is rejected";
      ok = !(builtins.tryEval (builtins.deepSeq
        (eval { services.djinn = { enable = true; address = "nope"; }; }).config.services.djinn.address true)).success; }
    { name = "default model renders the golden config";
      ok = minimal.config.services.djinn.modelConfig == golden; }
    { name = "overridden Hugging Face source is rendered";
      ok = custom.config.services.djinn.modelConfig.model_source.hugging_face_hub
        == { revision = "main"; repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
             weights_file = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
             tokenizer_repo = "Qwen/Qwen2.5-0.5B-Instruct"; }; }
    { name = "local files source drops the Hugging Face section";
      ok = localFiles.config.services.djinn.modelConfig.model_source
        == { files = { weight_files = [ "/models/q.gguf" ]; tokenizer_file = "/models/tokenizer.json"; }; }; }
    { name = "both sources set is an assertion failure";
      ok = failsWith bothSources "services.djinn.model.source"; }
    { name = "default is CPU: no device entry, private devices, no DeviceAllow";
      ok = !(minimal.config.services.djinn.modelConfig ? device)
        && (sc minimal).PrivateDevices && !(sc minimal ? DeviceAllow); }
    { name = "cuda.enable: CUDA package, device = Cuda, device nodes allowed";
      ok = has "${cuda}/bin/djinn-cli" (exec gpu)
        && gpu.config.services.djinn.modelConfig.device == "Cuda"
        && !(sc gpu).PrivateDevices
        && lib.any (has "nvidia") (sc gpu).DeviceAllow; }
    { name = "an explicit package wins over cuda.enable";
      ok = has "${cpu}/bin/djinn-cli" (exec gpuOwnPackage); }
    { name = "cuda.enable without graphics drivers is an assertion failure";
      ok = failsWith gpuNoDriver "hardware.graphics.enable"; }
    { name = "environment: RUST_LOG, HF_HOME under the state directory, extras merged";
      ok = let env = (service custom).environment; in
        env.FOO == "bar" && env.HF_HOME == "/var/lib/djinn/huggingface" && env.RUST_LOG == "debug"; }
    { name = "reserved variables cannot be overridden";
      ok = failsWith reservedEnv "services.djinn.extraEnvironment"; }
    { name = "environmentFile is passed to systemd as given";
      ok = (sc custom).EnvironmentFile == "/run/secrets/djinn.env"
        && !(sc minimal ? EnvironmentFile); }
    { name = "runtime: dynamic user, state directory, restart on failure";
      ok = (sc minimal).DynamicUser && (sc minimal).StateDirectory == "djinn"
        && (sc minimal).Restart == "on-failure"
        && (service minimal).after == [ "network-online.target" ]; }
    { name = "hardened by default";
      ok = (sc minimal).NoNewPrivileges && (sc minimal).PrivateTmp
        && (sc minimal).ProtectSystem == "strict" && (sc minimal).ProtectHome; }
  ];

  failures = map (t: t.name) (lib.filter (t: !t.ok) tests);
in
if failures == [ ] then
  pkgs.runCommand "nixos-module-check" { } "touch $out"
else
  throw "nixos-module check failed:\n  - ${lib.concatStringsSep "\n  - " failures}"
