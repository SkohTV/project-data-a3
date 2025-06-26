let filterData = {};
let currentPagination = 1;
let itemsPerPage = 10;
let totalPages = 1;
let totalCount = 0;

document.addEventListener("DOMContentLoaded", function () {
  loadFilterValues();
  loadtab();
  setupEventListeners();
});

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("add-vessel-form");
  const messageBox = document.getElementById("vessel-form-message");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const params = new URLSearchParams(formData);

    try {
      const response = await fetch("php/requests.php/add_boat", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: params.toString(),
      });

      if (response.status === 201) {
        messageBox.textContent = "Navire ajouté avec succès.";
        messageBox.style.color = "green";
        form.reset();
      } else if (response.status === 400) {
        messageBox.textContent = "Champs requis manquants.";
        messageBox.style.color = "red";
      } else if (response.status === 500) {
        messageBox.textContent = "Erreur serveur. Veuillez réessayer.";
        messageBox.style.color = "red";
      } else {
        messageBox.textContent = "Erreur inconnue.";
        messageBox.style.color = "red";
      }
    } catch (error) {
      messageBox.textContent = "Erreur de connexion au serveur.";
      messageBox.style.color = "red";
      console.error("Erreur fetch:", error);
    }
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("add-point-form");
  const messageBox = document.getElementById("point-form-message");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const params = new URLSearchParams(formData);

    try {
      const clusterUrl = `php/requests.php/predict_boat_cluster?latitude=${params.get(
        "latitude"
      )}&longitude=${params.get("longitude")}&sog=${params.get(
        "sog"
      )}&cog=${params.get("cog")}&heading=${params.get("heading")}`;

      const clusterResponse = await fetch(clusterUrl);

      if (!clusterResponse.ok) {
        throw new Error(`Cluster prediction failed: ${clusterResponse.status}`);
      }

      const clusterData = await clusterResponse.json();
      const id_cluster = clusterData[0];
      console.log("ClusterData:", clusterData);

      params.append("id_cluster", id_cluster);

      const response = await fetch("php/requests.php/add_point_donnee", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: params.toString(),
      });

      console.log("Paramètres envoyés:", params.toString());

      if (response.status === 201) {
        messageBox.textContent = "Point ajouté avec succès.";
        messageBox.style.color = "green";
        form.reset();
      } else if (response.status === 400) {
        messageBox.textContent = "Champs requis manquants.";
        messageBox.style.color = "red";
      } else if (response.status === 500) {
        messageBox.textContent = "Erreur serveur. Veuillez réessayer.";
        messageBox.style.color = "red";
      } else {
        messageBox.textContent = "Erreur inconnue.";
        messageBox.style.color = "red";
      }
    } catch (error) {
      console.error("Erreur:", error);
      messageBox.textContent =
        "Erreur de connexion au serveur ou prédiction du cluster.";
      messageBox.style.color = "red";
    }
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const vesselModeBtn = document.getElementById("vessel-mode-btn");
  const pointModeBtn = document.getElementById("point-mode-btn");
  const vesselSection = document.getElementById("vessel-section");
  const pointSection = document.getElementById("point-section");

  function switchMode(mode) {
    vesselModeBtn.classList.remove("active");
    pointModeBtn.classList.remove("active");

    vesselSection.classList.remove("active");
    pointSection.classList.remove("active");

    if (mode === "vessel") {
      vesselModeBtn.classList.add("active");
      vesselSection.classList.add("active");
    } else if (mode === "point") {
      pointModeBtn.classList.add("active");
      pointSection.classList.add("active");
    }

    vesselFormMessage.style.display = "none";
    pointFormMessage.style.display = "none";
    vesselFormMessage.className = "form-message";
    pointFormMessage.className = "form-message";
  }

  vesselModeBtn.addEventListener("click", () => switchMode("vessel"));
  pointModeBtn.addEventListener("click", () => switchMode("point"));
});

function loadFilterValues() {
  ajaxRequest("GET", "php/requests.php/get_filter_values", function (response) {
    filterData = response;
    initializeFilters();
  });
}

function initializeFilters() {
  setupSlider("longueur", filterData.longueur[1], filterData.longueur[0]);
  setupSlider("largeur", filterData.largeur[1], filterData.largeur[0]);
  setupDateSlider();
  populateSelect("filter-transceiver", filterData.transceiver, "Tous");
  populateSelect("filter-status", filterData.status_code, "Tous les status");
}

function setupSlider(type, min, max) {
  const minSlider = document.getElementById(`filter-${type}-min`);
  const maxSlider = document.getElementById(`filter-${type}-max`);
  const minVal = document.getElementById(`${type}-min-val`);
  const maxVal = document.getElementById(`${type}-max-val`);

  minSlider.min = minSlider.value = min;
  minSlider.max = max;
  maxSlider.min = min;
  maxSlider.max = maxSlider.value = max;
  minVal.textContent = min;
  maxVal.textContent = max;

  [minSlider, maxSlider].forEach((slider) => {
    slider.oninput = () => {
      minVal.textContent = minSlider.value;
      maxVal.textContent = maxSlider.value;
    };
  });
}

// date slider avec conversion en timestamp car sinon on ne peut pas faire de comparaison
function setupDateSlider() {
  const minSlider = document.getElementById("filter-temps-min");
  const maxSlider = document.getElementById("filter-temps-max");
  const minVal = document.getElementById("temps-min-val");
  const maxVal = document.getElementById("temps-max-val");

  minSlider.value = 0;
  maxSlider.value = 100;

  const updateDates = () => {
    const minDate = new Date(filterData.temps[1]);
    const maxDate = new Date(filterData.temps[0]);
    const range = maxDate.getTime() - minDate.getTime();

    const selectedMin = new Date(
      minDate.getTime() + (range * minSlider.value) / 100
    );
    const selectedMax = new Date(
      minDate.getTime() + (range * maxSlider.value) / 100
    );

    minVal.textContent = selectedMin.toLocaleDateString();
    maxVal.textContent = selectedMax.toLocaleDateString();
  };

  minSlider.oninput = maxSlider.oninput = updateDates;
  if (filterData.temps) updateDates();
}

function populateSelect(id, options, defaultText) {
  const select = document.getElementById(id);
  select.innerHTML = `<option value="">${defaultText}</option>`;
  options.forEach((option) => {
    select.innerHTML += `<option value="${option}">${option}</option>`;
  });
}

function showLoader() {
  const tbody = document.getElementById("vessels-tbody");
  tbody.innerHTML = `
    <tr>
      <td colspan="10" class="loader-cell">
        <div class="loader-container">
          <div class="loader-spinner"></div>
          <span class="loader-text">Chargement des données...</span>
        </div>
      </td>
    </tr>
  `;
}

function loadtab() {
  // FOR TAB
  showLoader();
  const mmsi = document.getElementById("filter-mmsi").value.trim();

  const params = new URLSearchParams({
    limits: itemsPerPage,
    page: currentPagination,
    longueur_min: document.getElementById("filter-longueur-min").value,
    longueur_max: document.getElementById("filter-longueur-max").value,
    largeur_min: document.getElementById("filter-largeur-min").value,
    largeur_max: document.getElementById("filter-largeur-max").value,
  });

  if (filterData.temps) {
    const minSliderValue = document.getElementById("filter-temps-min").value;
    const maxSliderValue = document.getElementById("filter-temps-max").value;

    const minDate = new Date(filterData.temps[1]);
    const maxDate = new Date(filterData.temps[0]);
    const range = maxDate.getTime() - minDate.getTime();

    const oneDay = 24 * 60 * 60 * 1000;

    const selectedMinDate = new Date(
      minDate.getTime() + (range * minSliderValue) / 100 - oneDay
    );
    const selectedMaxDate = new Date(
      minDate.getTime() + (range * maxSliderValue) / 100 + oneDay
    );

    params.append("temps_min", Math.floor(selectedMinDate.getTime() / 1000));
    params.append("temps_max", Math.floor(selectedMaxDate.getTime() / 1000));
  }

  if (mmsi) {
    params.append("mmsi", mmsi);
  }

  const transceiver = document.getElementById("filter-transceiver").value;
  const status = document.getElementById("filter-status").value;

  if (transceiver) {
    const transceiverCode =
      transceiver === "A" ? "1" : transceiver === "B" ? "2" : transceiver;
    params.append("transceiver_class", transceiverCode);
  }

  if (status) {
    params.append("status_code", status);
  }

  ajaxRequest("GET", `php/requests.php/get_tab?${params}`, function (response) {
    if (response.data) {
      displayData(response.data);
      updatePaginationInfo(response.pagination);
    } else {
      displayData(response);
      updatePagination(response.length);
    }
  });

  // FOR MAP
  ajaxRequest('GET', `php/requests.php/all_points_donnee?${params}`, (r) => {

    const transformed = r.reduce((acc, { mmsi, latitude, longitude }) => {

      if (!acc[mmsi])
        acc[mmsi] = { mmsi, color: '#F00', vals: [] };

      acc[mmsi].vals.push([latitude, longitude]);
      return acc;

    }, {});

    const transformedArray = Object.values(transformed);

    map_visu = generate_map('visu')

    console.log(transformedArray);
  })
}

function displayData(data) {
  const tbody = document.getElementById("vessels-tbody");

  if (!data || data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10">Aucune donnée disponible</td></tr>';
    return;
  }

  tbody.innerHTML = data
    .map(
      (point) => `
    <tr>
      <td><input type="radio" data-mmsi="${point.mmsi}" name="vessel"></td>
      <td>${point.mmsi}</td>
      <td>${new Date(point.base_date_time).toLocaleString()}</td>
      <td>${parseFloat(point.latitude).toFixed(6)}</td>
      <td>${parseFloat(point.longitude).toFixed(6)}</td>
      <td>${point.sog}</td>
      <td>${point.cog}</td>
      <td>${point.heading}</td>
      <td>${point.status_code}</td>
      <td>${point.draft}</td>
      <td>${point.transceiver ? "A" : "B"}</td>
      <td>${point.length}</td>
      <td>${point.width}</td>
    </tr>
  `
    )
    .join("");
}

function updatePaginationInfo(pagination) {
  totalPages = pagination.total_pages;
  totalCount = pagination.total_count;
  currentPagination = pagination.current_page;

  document.getElementById("prev-page").disabled = currentPagination <= 1;
  document.getElementById("next-page").disabled =
    currentPagination >= totalPages;

  document.getElementById(
    "pagination-info-text"
  ).textContent = `Page ${currentPagination} / ${totalPages} (${totalCount} résultat${
    totalCount > 1 ? "s" : ""
  } au total)`;
}

function updatePagination(dataLength) {
  const hasMore = dataLength === itemsPerPage;
  document.getElementById("prev-page").disabled = currentPagination <= 1;
  document.getElementById("next-page").disabled = !hasMore;
  document.getElementById(
    "pagination-info-text"
  ).textContent = `Page ${currentPagination}`;
}

function setupEventListeners() {
  document.getElementById("filter-button").onclick = () => {
    currentPagination = 1;
    loadtab();
  };

  document.getElementById("reset-button").onclick = resetFilters;

  document.getElementById("prev-page").onclick = () => {
    if (currentPagination > 1) {
      currentPagination--;
      loadtab();
    }
  };

  document.getElementById("next-page").onclick = () => {
    if (currentPagination < totalPages) {
      currentPagination++;
      loadtab();
    }
  };

  document.getElementById("items-per-page").onchange = (e) => {
    itemsPerPage = parseInt(e.target.value);
    currentPagination = 1;
    loadtab();
  };

  let timeout;
  document.getElementById("filter-mmsi").oninput = () => {
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      currentPagination = 1;
      loadtab();
    }, 500);
  };
}

function resetFilters() {
  if (!filterData.longueur) return;

  document.getElementById("filter-longueur-min").value = filterData.longueur[1];
  document.getElementById("filter-longueur-max").value = filterData.longueur[0];
  document.getElementById("filter-largeur-min").value = filterData.largeur[1];
  document.getElementById("filter-largeur-max").value = filterData.largeur[0];
  document.getElementById("filter-temps-min").value = 0;
  document.getElementById("filter-temps-max").value = 100;
  document.getElementById("filter-transceiver").value = "";
  document.getElementById("filter-status").value = "";
  document.getElementById("filter-mmsi").value = "";

  initializeFilters();
  currentPagination = 1;
  loadtab();
}

function showMessage(message) {
  document.getElementById(
    "vessels-tbody"
  ).innerHTML = `<tr><td colspan="10">${message}</td></tr>`;
}

function applyFilters() {
  loadtab();
}
function refreshVisualization() {
  loadtab();
}
function predictClusters() {
  alert("Je fais ca demain la team");
}
function predictSelected(type) {
  alert("Je fais ca demain la team, ou ce soir si j'ai le temps ");
}
function exportData() {
  alert("Je fais ca demain la team");
}
