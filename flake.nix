{

outputs = { self, nixpkgs }:

let
  system = "x86_64-linux";
  pkgs = nixpkgs.legacyPackages.${system};

  electif = (import ./electif/flake.nix).outputs {
    inherit system;
    inherit pkgs;
  };

in {

  devShells.${system} = {
    "electif" = electif.devShells.${system}.default; 
  };

};
}
