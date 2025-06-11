{

outputs = { pkgs, system }:

{
  devShells.${system}.default = pkgs.mkShell {
    packages = with pkgs; [
      R

      (with rPackages; [
        languageserver

        # r package
        ggplot2
        sf
        rnaturalearth
        rnaturalearthdata
        glm2
        leaflet
        devtools
        webshot
        mapview
      ])
    ];
  };

};
}
