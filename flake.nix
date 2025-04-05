{
  description = "Description for the project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{
    nixpkgs
    , flake-utils
    , crate2nix
    , rust-overlay
    , ...
    }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
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
        } ## /perSystem
    );
}
