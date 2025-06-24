function applyFilters() {
  const filters = getSelectedFilters();
  console.log("Filtres appliqués:", filters);

  const params = new URLSearchParams();

  params.append(
    "longueur_min",
    filters.longueur?.min ?? (typeof filterData !== "undefined" ? filterData.longueur[1] : "")
  );
  params.append(
    "longueur_max",
    filters.longueur?.max ?? (typeof filterData !== "undefined" ? filterData.longueur[0] : "")
  );

  params.append("largeur_min", filters.largeur?.min ?? (typeof filterData !== "undefined" ? filterData.largeur[1] : ""));
  params.append("largeur_max", filters.largeur?.max ?? (typeof filterData !== "undefined" ? filterData.largeur[0] : ""));

  if (filters.temps) {
    params.append("temps_min", Date.parse(filters.temps.min));
    params.append("temps_max", Date.parse(filters.temps.max));
  } else if (typeof dateRange !== "undefined") {
    params.append("temps_min", dateRange[0]);
    params.append("temps_max", dateRange[1]);
  } else {
    params.append("temps_min", "");
    params.append("temps_max", "");
  }

  params.append("transceiver_class", filters.transceiver ?? "");

  params.append("status_code", filters.status_code ?? "");
  if (typeof ajaxRequest === "function") {
    ajaxRequest(
      "GET",
      "php/all_points_donnee?" + params.toString(),
      function (response) {
        console.log(response);
      }
    );
  } else {
    console.error("ajaxRequest function is not defined.");
  }
}
