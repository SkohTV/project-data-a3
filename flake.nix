{

outputs = { self, nixpkgs }:

let
  system = "x86_64-linux";
  pkgs = nixpkgs.legacyPackages.${system};

  electif = (import ./electif/flake.nix).outputs { inherit system pkgs; };

in {

  devShells.${system} = {
    "electif" = electif.devShells.${system}.default; 
    "big-data" = throw "No devshell for this part of the project, didn't bother packaging it with Nix";
  };

};
}
