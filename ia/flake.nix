{

outputs = { pkgs, system }:

let
  python = pkgs.python312;
  latex = pkgs.texliveSmall;

in {
  devShells.${system}.default = pkgs.mkShell {
    packages = with pkgs; [
      pandoc

      (latex.withPackages (ps: with ps; [
        tcolorbox
        pdfcol
        adjustbox
        titling
        enumitem
        soul
        rsfs
      ]))

      (python.withPackages (ps: with ps; [
        virtualenv
        pip

        pandas 
        numpy
        scipy
        matplotlib
        seaborn
        scikit-learn
        notebook
        openpyxl
        pyarrow
        tensorflow
        keras
        joblib
        imbalanced-learn
        statsmodels
        lightgbm
        xgboost
      ]))
    ];
  };

};
}
