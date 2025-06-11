{

outputs = { self, nixpkgs }:

let
  system = "x86_64-linux";
  pkgs = nixpkgs.legacyPackages.${system};

  electif = (import ./electif/flake.nix).outputs { inherit system pkgs; };
  big-data = (import ./big-data/flake.nix).outputs { inherit system pkgs; };

in {

  devShells.${system} = {
    "electif" = electif.devShells.${system}.default; 
    "big-data" = big-data.devShells.${system}.default; 
  };

};
}
