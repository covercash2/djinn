{
  description = "Description for the project";

  inputs = {
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
    };
    crate2nix.url = "github:nix-community/crate2nix";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{
    nixpkgs
    , flake-parts
    , crate2nix
    , rust-overlay
    , ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        # To import a flake module
        # 1. Add foo to inputs
        # 2. Add foo as a parameter to the outputs function
        # 3. Add here: foo.flakeModule

      ];
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          buildRustCrateForPkgs =
            crate:
            pkgs.buildRustCrate.override {
              rustc = pkgs.rust-bin.nightly.latest.default;
              cargo = pkgs.rust-bin.nightly.latest.default;
            };
          generatedCargoNix = inputs.crate2nix.tools.${system}.generatedCargoNix {
            name = "djinn";
            src = ./.;
          };
          cargoNix = import generatedCargoNix {
            inherit pkgs buildRustCrateForPkgs;
          };
        in
          rec {
            # Per-system attributes can be defined here. The self' and inputs'
            # module parameters provide easy access to attributes of the same
            # system.

            checks = {
              djinn = cargoNix.workspaceMembers.djinn-cli.build.override {
                runTests = true;
              };
            };

            # Equivalent to  inputs'.nixpkgs.legacyPackages.hello;
            #packages.default = cargoNix.workspaceMembers.djinn-cli.build;
            packages = {
              djinn = cargoNix.workspaceMembers.djinn-cli.build;
              default = packages.djinn;
            };
          }; ## /perSystem
      flake = {
        # The usual flake attributes can be defined here, including system-
        # agnostic ones like nixosModule and system-enumerating ones, although
        # those are more easily expressed in perSystem.
      };
    };
}
