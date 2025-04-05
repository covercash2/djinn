{
  description = "djinn - machine learning experiments with candle";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ self, nixpkgs, flake-utils, rust-overlay, crane, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Use stable Rust with clippy and rustfmt
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "clippy" "rustfmt" "rust-src" ];
        };

        # Get system-specific dependencies
        systemDependencies = if pkgs.stdenv.isDarwin then
          with pkgs; [
            # macOS specific
            darwin.apple_sdk.frameworks.Metal
            darwin.apple_sdk.frameworks.Foundation
            darwin.apple_sdk.frameworks.Accelerate
            darwin.apple_sdk.frameworks.CoreGraphics
            darwin.apple_sdk.frameworks.CoreVideo
            darwin.libobjc
          ] else if pkgs.stdenv.isLinux then
            with pkgs; [
              # Linux specific
              cudaPackages.cudatoolkit
              cudaPackages.libcublas
              cudaPackages.cuda_cudart
              linuxPackages.nvidia_x11
            ] else [];

        # Common dependencies across platforms
        commonDependencies = with pkgs; [
          pkg-config
          openssl.dev
          rustToolchain
        ];

        # Crane setup for Rust package building
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Environment variables
        envVars = if pkgs.stdenv.isDarwin then {
          DYLD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath systemDependencies}";
        } else if pkgs.stdenv.isLinux then {
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath systemDependencies}";
          CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
          CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
        } else {};

        # Filter source to only include necessary files
        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            (craneLib.filterCargoSources path type) ||
            pkgs.lib.hasInfix "/assets/" path;
        };

        # Build common args that apply to all crates
        commonArgs = {
          inherit src;

          buildInputs = commonDependencies ++ systemDependencies;
          nativeBuildInputs = with pkgs; [ rustToolchain pkg-config ];

          # Add platform-specific feature flags
          cargoExtraArgs = if pkgs.stdenv.isDarwin then
            "--features mac"
          else if pkgs.stdenv.isLinux then
            "--features cuda"
          else
            "";
        };

        # Build dependencies separately to improve caching
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Build the djinn-cli package (the main binary)
        djinn-cli = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          pname = "djinn-cli";
          cargoExtraArgs = commonArgs.cargoExtraArgs + " -p djinn-cli";
        });

        # Build the ollama-cli package
        ollama-cli = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          pname = "ollama-cli";
          cargoExtraArgs = commonArgs.cargoExtraArgs + " -p ollama-cli";
        });
      in
      {
        # Use djinn-cli as the default package
        packages = {
          djinn-cli = djinn-cli;
          ollama-cli = ollama-cli;
          default = djinn-cli;
        };

        # Add other checks and test cases
        checks = {
          djinn-tests = craneLib.cargoTest (commonArgs // {
            inherit cargoArtifacts;
          });
        };

        # Development shell with all dependencies
        devShells.default = pkgs.mkShell {
          inputsFrom = [ djinn-cli ];
          packages = with pkgs; [
            rustToolchain
            rust-analyzer
            cargo-watch
            cargo-expand
            cargo-audit
          ];

          # Include environment variables for development
          inherit envVars;
          shellHook = ''
            ${pkgs.lib.concatStringsSep "\n" (pkgs.lib.mapAttrsToList (k: v: "export ${k}=${v}") envVars)}
            echo "djinn development environment loaded"
          '';
        };
      }
    );
}

