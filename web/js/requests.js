let filterData = {};
let dateRange = [];

let currentPage = 1;
let itemsPerPage = 25;
let totalItems = 0;
let currentFilters = {};

document.addEventListener("DOMContentLoaded", function () {
  loadFilterValues();
  loadtab();
});

function loadFilterValues() {
  ajaxRequest("GET", "php/requests.php/get_filter_values", function (response) {
    filterData = response;
    initializeFilters();
  });
}

function initializeFilters() {
  setupRangeSlider("longueur", filterData.longueur[1], filterData.longueur[0]);
  setupRangeSlider("largeur", filterData.largeur[1], filterData.largeur[0]);

  const minDate = new Date(filterData.temps[1]); // temps en timestamp car sinon on ne peut pas faire un slider
  const maxDate = new Date(filterData.temps[0]);
  dateRange = [minDate.getTime(), maxDate.getTime()];
  setupDateRangeSlider(minDate.getTime(), maxDate.getTime());

  const transceiverSelect = document.getElementById("filter-transceiver");
  transceiverSelect.innerHTML = '<option value="">Tous</option>';
  filterData.transceiver.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    transceiverSelect.appendChild(option);
  });

  const statusSelect = document.getElementById("filter-status");
  statusSelect.innerHTML = '<option value="">Tous les status</option>';
  filterData.status_code.forEach((code) => {
    const option = document.createElement("option");
    option.value = code;
    option.textContent = code;
    statusSelect.appendChild(option);
  });
}

function setupRangeSlider(type, min, max) {
  const minSlider = document.getElementById(`filter-${type}-min`);
  const maxSlider = document.getElementById(`filter-${type}-max`);
  const minVal = document.getElementById(`${type}-min-val`);
  const maxVal = document.getElementById(`${type}-max-val`);

  minSlider.min = min;
  minSlider.max = max;
  minSlider.value = min;

  maxSlider.min = min;
  maxSlider.max = max;
  maxSlider.value = max;

  minVal.textContent = min;
  maxVal.textContent = max;

  updateRangeTrack(type);

  minSlider.addEventListener("input", function () {
    if (parseInt(this.value) > parseInt(maxSlider.value)) {
      this.value = maxSlider.value;
    }
    minVal.textContent = this.value;
    updateRangeTrack(type);
  });

  maxSlider.addEventListener("input", function () {
    if (parseInt(this.value) < parseInt(minSlider.value)) {
      this.value = minSlider.value;
    }
    maxVal.textContent = this.value;
    updateRangeTrack(type);
  });
}

function setupDateRangeSlider(minTimestamp, maxTimestamp) {
  const minSlider = document.getElementById("filter-temps-min");
  const maxSlider = document.getElementById("filter-temps-max");
  const minVal = document.getElementById("temps-min-val");
  const maxVal = document.getElementById("temps-max-val");

  minSlider.min = 0;
  minSlider.max = 100;
  minSlider.value = 0;

  maxSlider.min = 0;
  maxSlider.max = 100;
  maxSlider.value = 100;

  function updateDateDisplay() {
    const minPercent = parseInt(minSlider.value);
    const maxPercent = parseInt(maxSlider.value);

    const minDate = new Date(
      minTimestamp + ((maxTimestamp - minTimestamp) * minPercent) / 100
    );
    const maxDate = new Date(
      minTimestamp + ((maxTimestamp - minTimestamp) * maxPercent) / 100
    );

    minVal.textContent = formatDate(minDate);
    maxVal.textContent = formatDate(maxDate);

    updateRangeTrack("temps");
  }

  updateDateDisplay();

  minSlider.addEventListener("input", function () {
    if (parseInt(this.value) > parseInt(maxSlider.value)) {
      this.value = maxSlider.value;
    }
    updateDateDisplay();
  });

  maxSlider.addEventListener("input", function () {
    if (parseInt(this.value) < parseInt(minSlider.value)) {
      this.value = minSlider.value;
    }
    updateDateDisplay();
  });
}

function formatDate(date) {
  const day = String(date.getDate()).padStart(2, "0");
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const year = date.getFullYear();
  return `${day}/${month}/${year}`;
}

function updateRangeTrack(type) {
  const minSlider = document.getElementById(`filter-${type}-min`);
  const maxSlider = document.getElementById(`filter-${type}-max`);
  const track = minSlider.parentElement.querySelector(".range-track");

  const min = parseInt(minSlider.min);
  const max = parseInt(minSlider.max);
  const minVal = parseInt(minSlider.value);
  const maxVal = parseInt(maxSlider.value);

  const leftPercent = ((minVal - min) / (max - min)) * 100;
  const rightPercent = ((maxVal - min) / (max - min)) * 100;

  track.style.background = `linear-gradient(to right, 
        rgba(255, 255, 255, 0.3) 0%, 
        rgba(255, 255, 255, 0.3) ${leftPercent}%, 
        #64b5f6 ${leftPercent}%, 
        #64b5f6 ${rightPercent}%, 
        rgba(255, 255, 255, 0.3) ${rightPercent}%, 
        rgba(255, 255, 255, 0.3) 100%)`;
}

function getSelectedFilters() {
  const filters = {};

  const longueurMin = document.getElementById("filter-longueur-min").value;
  const longueurMax = document.getElementById("filter-longueur-max").value;
  if (
    longueurMin !== filterData.longueur[1] ||
    longueurMax !== filterData.longueur[0]
  ) {
    filters.longueur = { min: longueurMin, max: longueurMax };
  }

  const largeurMin = document.getElementById("filter-largeur-min").value;
  const largeurMax = document.getElementById("filter-largeur-max").value;
  if (
    largeurMin !== filterData.largeur[1] ||
    largeurMax !== filterData.largeur[0]
  ) {
    filters.largeur = { min: largeurMin, max: largeurMax };
  }

  const tempsMinPercent = parseInt(
    document.getElementById("filter-temps-min").value
  );
  const tempsMaxPercent = parseInt(
    document.getElementById("filter-temps-max").value
  );
  if (tempsMinPercent !== 0 || tempsMaxPercent !== 100) {
    const minTimestamp =
      dateRange[0] + ((dateRange[1] - dateRange[0]) * tempsMinPercent) / 100;
    const maxTimestamp =
      dateRange[0] + ((dateRange[1] - dateRange[0]) * tempsMaxPercent) / 100;
    filters.temps = {
      min: new Date(minTimestamp).toISOString(),
      max: new Date(maxTimestamp).toISOString(),
    };
  }

  const transceiver = document.getElementById("filter-transceiver").value;
  if (transceiver) {
    filters.transceiver = transceiver;
  }

  const mmsi = document.getElementById("filter-mmsi").value.trim();
  if (mmsi) {
    filters.mmsi = mmsi;
  }

  const statusCode = document.getElementById("filter-status").value;
  if (statusCode) {
    filters.status_code = statusCode;
  }

  return filters;
}

function applyFilters() {
  const filters = getSelectedFilters();
  console.log("Filtres appliqués:", filters);

  // ajaxRequest('POST', 'php/filter_vessels', function(response) {
  //     updateVesselsTable(response);
  // }, JSON.stringify(filters));
}

function resetFilters() {
  if (!filterData.longueur) {
    console.log("Les données de filtres ne sont pas encore chargées");
    return;
  }

  document.getElementById("filter-longueur-min").value = filterData.longueur[1];
  document.getElementById("filter-longueur-max").value = filterData.longueur[0];
  document.getElementById("longueur-min-val").textContent =
    filterData.longueur[1];
  document.getElementById("longueur-max-val").textContent =
    filterData.longueur[0];
  updateRangeTrack("longueur");

  document.getElementById("filter-largeur-min").value = filterData.largeur[1];
  document.getElementById("filter-largeur-max").value = filterData.largeur[0];
  document.getElementById("largeur-min-val").textContent =
    filterData.largeur[1];
  document.getElementById("largeur-max-val").textContent =
    filterData.largeur[0];
  updateRangeTrack("largeur");

  document.getElementById("filter-temps-min").value = 0;
  document.getElementById("filter-temps-max").value = 100;
  const minDate = new Date(dateRange[0]);
  const maxDate = new Date(dateRange[1]);
  document.getElementById("temps-min-val").textContent = formatDate(minDate);
  document.getElementById("temps-max-val").textContent = formatDate(maxDate);
  updateRangeTrack("temps");

  document.getElementById("filter-transceiver").value = "";
  document.getElementById("filter-status").value = "";

  document.getElementById("filter-mmsi").value = "";

  console.log("Filtres réinitialisés");
}

function loadtab(filters = null, page = 1, limits = null) {
  currentPage = page;
  if (limits) itemsPerPage = limits;
  if (filters) currentFilters = filters;

  const params = new URLSearchParams();
  params.append("limits", itemsPerPage);
  params.append("page", currentPage);

  if (currentFilters.mmsi) {
    params.append("mmsi", currentFilters.mmsi);
  }
  if (currentFilters.longueur) {
    params.append("longueur_min", currentFilters.longueur.min);
    params.append("longueur_max", currentFilters.longueur.max);
  }
  if (currentFilters.largeur) {
    params.append("largeur_min", currentFilters.largeur.min);
    params.append("largeur_max", currentFilters.largeur.max);
  }
  if (currentFilters.temps) {
    params.append("temps_min", currentFilters.temps.min);
    params.append("temps_max", currentFilters.temps.max);
  }
  if (currentFilters.transceiver) {
    params.append("transceiver_class", currentFilters.transceiver);
  }
  if (currentFilters.status_code) {
    params.append("status_code", currentFilters.status_code);
  }

  const url = `php/requests.php/get_tab?${params.toString()}`;

  ajaxRequest("GET", url, function (response) {
    const tableBody = document.getElementById("vessels-tbody");
    tableBody.innerHTML = "";

    if (response && response.length > 0) {
      response.forEach((point) => {
        const row = document.createElement("tr");
        row.innerHTML = `
                    <td><input type="checkbox" class="vessel-checkbox" data-mmsi="${
                      point.mmsi
                    }" data-timestamp="${point.base_date_time}"></td>
                    <td>${point.mmsi || "N/A"}</td>
                    <td>${
                      point.base_date_time
                        ? new Date(point.base_date_time).toLocaleString()
                        : "N/A"
                    }</td>
                    <td>${point.latitude || "N/A"}</td>
                    <td>${point.longitude || "N/A"}</td>
                    <td>${point.sog || "N/A"}</td>
                    <td>${point.cog || "N/A"}</td>
                    <td>${point.heading || "N/A"}</td>
                    <td>${point.status_code || "N/A"}</td>
                    <td>${point.draft || "N/A"}</td>
                `;
        tableBody.appendChild(row);
      });

      totalItems =
        response.length === itemsPerPage
          ? currentPage * itemsPerPage + 1
          : (currentPage - 1) * itemsPerPage + response.length;
    } else {
      const row = document.createElement("tr");
      row.innerHTML = `<td colspan="10" class="placeholder-text">Aucune donnée disponible</td>`;
      tableBody.appendChild(row);
      totalItems = 0;
    }

    updatePaginationInfo();
    updatePaginationControls();
  });

  initializeEventListeners();
}

function updatePaginationInfo() {
  const totalPages = Math.ceil(totalItems / itemsPerPage);
  const startItem = totalItems > 0 ? (currentPage - 1) * itemsPerPage + 1 : 0;
  const endItem = Math.min(currentPage * itemsPerPage, totalItems);

  const infoText = `Page ${currentPage} sur ${totalPages} (${totalItems} résultats)`;
  document.getElementById("pagination-info-text").textContent = infoText;
}

function updatePaginationControls() {
  const totalPages = Math.ceil(totalItems / itemsPerPage);

  const prevButton = document.getElementById("prev-page");
  const nextButton = document.getElementById("next-page");

  prevButton.disabled = currentPage <= 1;
  nextButton.disabled = currentPage >= totalPages || totalItems === 0;

  const pageNumbers = document.getElementById("page-numbers");
  pageNumbers.innerHTML = "";

  if (totalPages > 1) {
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);

    for (let i = startPage; i <= endPage; i++) {
      const pageButton = document.createElement("button");
      pageButton.textContent = i;
      pageButton.className =
        i === currentPage ? "btn-primary" : "btn-secondary";
      pageButton.style.margin = "0 2px";
      pageButton.onclick = () => loadtab(currentFilters, i, itemsPerPage);
      pageNumbers.appendChild(pageButton);
    }
  }
}

function initializeEventListeners() {
  const itemsPerPageSelect = document.getElementById("items-per-page");
  if (
    itemsPerPageSelect &&
    !itemsPerPageSelect.hasAttribute("data-listener-attached")
  ) {
    itemsPerPageSelect.addEventListener("change", function () {
      itemsPerPage = parseInt(this.value);
      currentPage = 1;
      loadtab(currentFilters, currentPage, itemsPerPage);
    });
    itemsPerPageSelect.setAttribute("data-listener-attached", "true");
  }

  const prevButton = document.getElementById("prev-page");
  if (prevButton && !prevButton.hasAttribute("data-listener-attached")) {
    prevButton.addEventListener("click", function () {
      if (currentPage > 1) {
        loadtab(currentFilters, currentPage - 1, itemsPerPage);
      }
    });
    prevButton.setAttribute("data-listener-attached", "true");
  }

  const nextButton = document.getElementById("next-page");
  if (nextButton && !nextButton.hasAttribute("data-listener-attached")) {
    nextButton.addEventListener("click", function () {
      const totalPages = Math.ceil(totalItems / itemsPerPage);
      if (currentPage < totalPages) {
        loadtab(currentFilters, currentPage + 1, itemsPerPage);
      }
    });
    nextButton.setAttribute("data-listener-attached", "true");
  }

  const filterButton = document.getElementById("filter-button");
  if (filterButton && !filterButton.hasAttribute("data-listener-attached")) {
    filterButton.addEventListener("click", function () {
      const filters = getSelectedFilters();
      currentPage = 1;
      loadtab(filters, currentPage, itemsPerPage);
    });
    filterButton.setAttribute("data-listener-attached", "true");
  }

  const resetButton = document.getElementById("reset-button");
  if (resetButton && !resetButton.hasAttribute("data-listener-attached")) {
    resetButton.addEventListener("click", function () {
      resetFilters();
      currentFilters = {};
      currentPage = 1;
      loadtab(null, currentPage, itemsPerPage);
    });
    resetButton.setAttribute("data-listener-attached", "true");
  }

  const mmsiInput = document.getElementById("filter-mmsi");
  if (mmsiInput && !mmsiInput.hasAttribute("data-listener-attached")) {
    let timeout;
    mmsiInput.addEventListener("input", function () {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        const filters = getSelectedFilters();
        if (this.value.trim() === "") {
          delete filters.mmsi;
        } else {
          filters.mmsi = this.value.trim();
        }
        currentPage = 1;
        loadtab(filters, currentPage, itemsPerPage);
      }, 300);
    });
    mmsiInput.setAttribute("data-listener-attached", "true");
  }
}

function applyFilters() {
  const filters = getSelectedFilters();
  currentPage = 1;
  loadtab(filters, currentPage, itemsPerPage);
}

function refreshVisualization() {
  loadtab(currentFilters, currentPage, itemsPerPage);
}
