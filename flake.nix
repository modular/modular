{
  description = "Modular Platform (Mojo compiler, standard library and MAX) — reproducible, offline Nix build of the release";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        modular = pkgs.callPackage ./nix { inherit self; };
      in
      {
        inherit (modular) packages devShells checks;
      }
    )
    // {
      overlays.default = final: prev: {
        modular = final.callPackage ./nix { inherit self; };
        mojo = final.modular.packages.mojo;
      };
    };
}
