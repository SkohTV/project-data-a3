{

outputs = { pkgs, system }:

{
  devShells.${system}.default = pkgs.mkShell {
    packages = with pkgs; [
      R

      (with rPackages; [
        languageserver

        # r package
      ])
    ];
  };

};
}
