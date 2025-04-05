# djinn - Nix Flake Configuration
# ===============================
#
# This flake provides a development environment and build setup for the `djinn` project,
# supporting both macOS (using Metal) and Linux (using CUDA) platforms.
#
# REFERENCES:
# - Nix Flakes: https://nixos.wiki/wiki/Flakes
# - Rust & Nix: https://nixos.org/manual/nixpkgs/stable/#rust
# - Crane: https://github.com/ipetkov/crane
# - Rust Overlay: https://github.com/oxalica/rust-overlay
#
# https://nixos.org/manual/nix/stable/command-ref/new-cli/nix3-flake.html
{
  description = "djinn - machine learning experiments with candle";

  # Define external dependencies for this flake
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

  # The main content of the flake - what it provides to users
  outputs = inputs@{ self, nixpkgs, flake-utils, rust-overlay, crane, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Define the Rust compiler and tools.
        # https://github.com/oxalica/rust-overlay#rust-binrust-toolchainstable
        rustToolchain = pkgs.rust-bin.nightly."2025-02-01".default.override {
          extensions = [ "clippy" "rustfmt" "rust-src" ];
        };

        # Get system-specific dependencies
        systemDependencies = if pkgs.stdenv.isDarwin then
          with pkgs; [
            # Metal is Apple's GPU computing framework
            darwin.apple_sdk.frameworks.Metal
            darwin.apple_sdk.frameworks.Foundation
            # Accelerate framework provides optimized math operations
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

        # Configure Crane for building Rust packages
        # https://crane.dev/API.html
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Set up environment variables based on the platform
        # https://nixos.org/manual/nix/stable/command-ref/conf-file.html#environment-variables
        shellEnvSetup = if pkgs.stdenv.isDarwin then ''
          # On macOS, set dynamic library path to find Metal and Accelerate frameworks
          export DYLD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath systemDependencies}
          echo "🍎 macOS environment: Metal acceleration enabled"
        '' else if pkgs.stdenv.isLinux then ''
          # On Linux, configure CUDA environment
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath systemDependencies}
          export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
          export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
          echo "🐧 Linux environment: CUDA acceleration enabled"
        '' else "";

        # Filter source to only include necessary files.
        # Only include needed files to improve build caching.
        # https://nixos.org/manual/nixpkgs/stable/#sec-functions-library-path-manipulation
        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            (craneLib.filterCargoSources path type) ||
            pkgs.lib.hasInfix "/assets/" path;
        };

        # Build common args that apply to all crates
        commonArgs = {
          inherit src;

          pname = "djinn";
          version = "0.1.0";

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

        # Build dependencies separately for better caching
        # https://crane.dev/API.html#cranelibbuildDepsOnly
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # PACKAGE DEFINITIONS
        # ==================
        # Define how to build each package in the workspace
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
        # Define the packages this flake provides
        # https://nixos.org/manual/nix/stable/command-ref/new-cli/nix3-build.html
        # Use djinn-cli as the default package
        packages = {
          djinn-cli = djinn-cli;
          ollama-cli = ollama-cli;
          default = djinn-cli;
        };

        # Add other checks and test cases
        # https://nixos.org/manual/nix/stable/command-ref/new-cli/nix3-flake-check.html
        checks = {
          djinn-tests = craneLib.cargoTest (commonArgs // {
            inherit cargoArtifacts;
          });
        };

        # Development shell with all dependencies
        # https://nixos.wiki/wiki/Development_environment_with_nix-shell
        # https://nixos.org/manual/nixpkgs/stable/#sec-pkgs-mkShell
        # https://nix.dev/tutorials/first-steps/declarative-shell.html
        devShells.default = pkgs.mkShell {
          inputsFrom = [ djinn-cli ];
          packages = with pkgs; [
            cargo-audit
            cargo-expand
            cargo-udeps
            cargo-watch
            just
            rust-analyzer
            rustToolchain
          ];

          # Include environment variables for development
          shellHook = ''
            ${shellEnvSetup}
            export RUST_LOG=info
            echo ""
            echo "🔧 djinn development environment loaded"
            echo ""
            echo "Available commands:"
            echo "  cargo build             - Build the project"
            echo "  cargo test              - Run tests"
            echo "  cargo run -p djinn-cli  - Run the djinn CLI"
            echo "  cargo run -p ollama-cli - Run the ollama CLI"
            echo ""
            echo "To build with Nix:"
            echo "  nix build               - Build the default package (djinn-cli)"
            echo "  nix build .#ollama-cli  - Build a specific package"
            echo ""
          '';
        };
      }
    );
}
