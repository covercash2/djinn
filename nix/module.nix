# NixOS module: `services.djinn`, the djinn server as a systemd service.
#
#   services.djinn = { enable = true; port = 9000; };
#
# The model is described with options and rendered to the TOML model config the server reads, so no
# hand-written file is needed. CUDA is opt-in (`cuda.enable`). Design: openspec nixos-service.
self:
{ config, lib, pkgs, ... }:

let
  inherit (lib) mkEnableOption mkIf mkOption types;

  cfg = config.services.djinn;
  system = pkgs.stdenv.hostPlatform.system;

  # Same defaults as the server's `ModelSource::HuggingFaceHub`.
  hfDefaults = {
    revision = "main";
    repo = null;
    weightsFile = null;
    tokenizerRepo = null;
    tokenizerRevision = null;
  };

  ipv4 = "([0-9]{1,3}\\.){3}[0-9]{1,3}";
  ipv6 = "[0-9a-fA-F:.]*:[0-9a-fA-F:.]*";
  ipAddress = types.addCheck types.str
    (s: builtins.match "(${ipv4}|${ipv6})" s != null) // {
      name = "ipAddress";
      description = "an IPv4 or IPv6 address";
    };

  hf = cfg.model.source.huggingFace;
  files = cfg.model.source.files;

  # The model config in the format `djinn_core::lm::config::ModelConfig` deserializes.
  nonNull = lib.filterAttrs (_: v: v != null);
  modelSource =
    if files != null then {
      files = {
        weight_files = files.weightFiles;
        tokenizer_file = files.tokenizerFile;
      };
    } else {
      hugging_face_hub = nonNull {
        revision = hf.revision;
        repo_id = hf.repo;
        weights_file = hf.weightsFile;
        tokenizer_repo = hf.tokenizerRepo;
        tokenizer_revision = hf.tokenizerRevision;
      };
    };
  modelConfig = nonNull {
    variant = cfg.model.architecture;
    # Left out on a CPU build so the server picks the best device compiled in.
    device = if cfg.cuda.enable then "Cuda" else null;
    model_source = modelSource;
  };
  modelConfigFile = (pkgs.formats.toml { }).generate "djinn-model.toml" modelConfig;
  # `djinn-cli server --config-dir <this> --model-config model/djinn`
  configDir = pkgs.runCommand "djinn-configs" { } ''
    mkdir -p $out/model
    cp ${modelConfigFile} $out/model/djinn.toml
  '';

  stateDir = "/var/lib/djinn";
in
{
  options.services.djinn = {
    enable = mkEnableOption "the djinn language model server";

    package = mkOption {
      type = types.package;
      default =
        self.packages.${system}.${if cfg.cuda.enable then "djinn-cli-cuda" else "djinn-cli-cpu"};
      defaultText = lib.literalMD
        "`djinn-cli-cpu` from the djinn flake, or `djinn-cli-cuda` when `cuda.enable` is set";
      description = ''
        The package providing `djinn-cli`. When set explicitly it is used as is: it must be a CUDA
        build if `cuda.enable` is true.
      '';
    };

    address = mkOption {
      type = ipAddress;
      default = "127.0.0.1";
      description = "IP address the server listens on.";
    };

    port = mkOption {
      type = types.port;
      default = 8080;
      description = "TCP port the server listens on.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Open `port` in the firewall.";
    };

    logLevel = mkOption {
      type = types.str;
      default = "warn,djinn_server=info,djinn_core=info";
      description = "Value of `RUST_LOG` (a `tracing` filter).";
    };

    extraEnvironment = mkOption {
      type = types.attrsOf types.str;
      default = { };
      description = "Extra environment variables. `HF_HOME` and `RUST_LOG` are set by the module.";
    };

    environmentFile = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "/run/secrets/djinn.env";
      description = ''
        Absolute path of a file of `NAME=value` lines read by systemd, for secrets such as a
        Hugging Face token. It is given as a string so it is never copied into the Nix store.
      '';
    };

    modelConfig = mkOption {
      type = types.attrs;
      readOnly = true;
      internal = true;
      default = modelConfig;
      description = "The model config the module renders to TOML (exposed for the flake check).";
    };

    cuda.enable = mkEnableOption ''
      CUDA acceleration: the CUDA build, `device = "Cuda"` in the model config and access to the
      GPU device nodes. Needs `hardware.graphics.enable` and the NVIDIA driver on the host'';

    model = {
      architecture = mkOption {
        type = types.enum [ "qwen2_5_instruct" ];
        default = "qwen2_5_instruct";
        description = "Model architecture.";
      };

      source.huggingFace = {
        revision = mkOption {
          type = types.str;
          default = hfDefaults.revision;
          description = "Hugging Face revision of the model repository.";
        };
        repo = mkOption {
          type = types.nullOr types.str;
          default = hfDefaults.repo;
          example = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
          description = "Repository with the weights (`owner/name`). Null uses the architecture's default.";
        };
        weightsFile = mkOption {
          type = types.nullOr types.str;
          default = hfDefaults.weightsFile;
          example = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
          description = "GGUF file to download. Null uses the architecture's default.";
        };
        tokenizerRepo = mkOption {
          type = types.nullOr types.str;
          default = hfDefaults.tokenizerRepo;
          description = "Repository to take `tokenizer.json` from, when it is not in `repo`.";
        };
        tokenizerRevision = mkOption {
          type = types.nullOr types.str;
          default = hfDefaults.tokenizerRevision;
          description = "Revision of the tokenizer repository. Null means `main`.";
        };
      };

      source.files = mkOption {
        type = types.nullOr (types.submodule {
          options = {
            weightFiles = mkOption {
              type = types.listOf types.str;
              description = "Absolute paths of the weight files.";
            };
            tokenizerFile = mkOption {
              type = types.str;
              description = "Absolute path of `tokenizer.json`.";
            };
          };
        });
        default = null;
        description = ''
          Load the model from local files instead of the Hugging Face Hub. Strings, not paths, so
          large weights are not copied into the Nix store. Excludes the `huggingFace` options.
        '';
      };
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = files == null || hf == hfDefaults;
        message = "services.djinn.model.source: set either `files` or the `huggingFace` options, not both.";
      }
      {
        assertion = !(cfg.extraEnvironment ? HF_HOME || cfg.extraEnvironment ? RUST_LOG);
        message = "services.djinn.extraEnvironment: HF_HOME and RUST_LOG are set by the module (use `logLevel`).";
      }
      {
        assertion = cfg.environmentFile == null || lib.hasPrefix "/" cfg.environmentFile;
        message = "services.djinn.environmentFile must be an absolute path.";
      }
      {
        assertion = !cfg.cuda.enable
          || lib.attrByPath [ "hardware" "graphics" "enable" ] false config;
        message = "services.djinn.cuda.enable needs `hardware.graphics.enable = true` (and the NVIDIA driver) so the host libcuda is available.";
      }
    ];

    networking.firewall.allowedTCPPorts = lib.optional cfg.openFirewall cfg.port;

    systemd.services.djinn = {
      description = "djinn language model server";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];

      environment = cfg.extraEnvironment // {
        HF_HOME = "${stateDir}/huggingface";
        RUST_LOG = cfg.logLevel;
      };

      serviceConfig = {
        ExecStart = lib.escapeShellArgs [
          "${cfg.package}/bin/djinn-cli"
          "server"
          "--ip"
          cfg.address
          "--port"
          (toString cfg.port)
          "--config-dir"
          "${configDir}"
          "--model-config"
          "model/djinn"
        ];
        Type = "simple";
        Restart = "on-failure";
        RestartSec = 5;
        # The first start downloads the weights (about 1 GB for the default model).
        TimeoutStartSec = "30min";

        DynamicUser = true;
        StateDirectory = "djinn";
        WorkingDirectory = stateDir;

        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectKernelLogs = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
        RestrictNamespaces = true;
        RestrictRealtime = true;
        LockPersonality = true;
        SystemCallArchitectures = "native";
      } // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      } // (if cfg.cuda.enable then {
        PrivateDevices = false;
        SupplementaryGroups = [ "video" "render" ];
        DeviceAllow = [
          "char-nvidia-frontend rw"
          "char-nvidia-uvm rw"
          "char-nvidia-caps rw"
          "char-nvidiactl rw"
          "char-drm rw"
        ];
      } else {
        PrivateDevices = true;
      });
    };
  };
}
