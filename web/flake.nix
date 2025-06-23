{

outputs = { pkgs, system }:

let
  python = pkgs.python312;
  latex = pkgs.texliveSmall;

in {
  devShells.${system}.default = pkgs.mkShell {
    packages = with pkgs; [
      pandoc

      (python.withPackages (ps: with ps; [
        virtualenv
        pip

        requests
      ]))
    ];
  };

};
}
