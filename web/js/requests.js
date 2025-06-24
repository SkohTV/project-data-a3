
let filterData = {};
let currentPagination = 1;
let itemsPerPage = 25;


document.addEventListener("DOMContentLoaded", function () {
  loadFilterValues();
  loadFirstMMSI();
  setupEventListeners();
});


function loadFilterValues() {
  ajaxRequest("GET", "php/requests.php/get_filter_values", function (response) {
    filterData = response;
    initializeFilters();
  });
}


function loadFirstMMSI() {
  ajaxRequest("GET", "php/requests.php/all_mmsi", function(response) {
    if (response && response.length > 0) {
      document.getElementById("filter-mmsi").value = response[0].mmsi;
      loadtab();
    }
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

  [minSlider, maxSlider].forEach(slider => {
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
    
    const selectedMin = new Date(minDate.getTime() + (range * minSlider.value / 100));
    const selectedMax = new Date(minDate.getTime() + (range * maxSlider.value / 100));
    
    minVal.textContent = selectedMin.toLocaleDateString();
    maxVal.textContent = selectedMax.toLocaleDateString();
  };

  minSlider.oninput = maxSlider.oninput = updateDates;
  if (filterData.temps) updateDates();
}


function populateSelect(id, options, defaultText) {
  const select = document.getElementById(id);
  select.innerHTML = `<option value="">${defaultText}</option>`;
  options.forEach(option => {
    select.innerHTML += `<option value="${option}">${option}</option>`;
  });
}


function loadtab() {
  const mmsi = document.getElementById("filter-mmsi").value.trim();
  if (!mmsi) {
    showMessage("Entrer un MMSI");
    return;
  }

  const params = new URLSearchParams({
    limits: itemsPerPage,
    page: currentPagination,
    mmsi: mmsi,
    longueur_min: document.getElementById("filter-longueur-min").value,
    longueur_max: document.getElementById("filter-longueur-max").value,
    largeur_min: document.getElementById("filter-largeur-min").value,
    largeur_max: document.getElementById("filter-largeur-max").value,
    temps_min: filterData.temps ? filterData.temps[1] : '2024-01-01 00:00:00',
    temps_max: filterData.temps ? filterData.temps[0] : '2024-12-31 23:59:59'
  });


  const transceiver = document.getElementById("filter-transceiver").value;
  const status = document.getElementById("filter-status").value;
  if (transceiver) params.append('transceiver_class', transceiver);
  if (status) params.append('status_code', status);

  ajaxRequest("GET", `php/requests.php/get_tab?${params}`, function(response) {
    displayData(response);
    updatePagination(response.length);
  });
}


function displayData(data) {
  const tbody = document.getElementById("vessels-tbody");
  
  if (!data || data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10">Aucune donnée disponible</td></tr>';
    return;
  }

  tbody.innerHTML = data.map(point => `
    <tr>
      <td><input type="checkbox" data-mmsi="${point.mmsi}"></td>
      <td>${point.mmsi}</td>
      <td>${new Date(point.base_date_time).toLocaleString()}</td>
      <td>${parseFloat(point.latitude).toFixed(6)}</td>
      <td>${parseFloat(point.longitude).toFixed(6)}</td>
      <td>${point.sog}</td>
      <td>${point.cog}</td>
      <td>${point.heading}</td>
      <td>${point.status_code}</td>
      <td>${point.draft}</td>
    </tr>
  `).join('');
}


function updatePagination(dataLength) {
  const hasMore = dataLength === itemsPerPage;
  document.getElementById('prev-page').disabled = currentPagination <= 1;
  document.getElementById('next-page').disabled = !hasMore;
  document.getElementById('pagination-info-text').textContent = `Page ${currentPagination}`;
}


function setupEventListeners() {

  document.getElementById("filter-button").onclick = () => {
    currentPagination = 1;
    loadtab();
  };
  
  document.getElementById("reset-button").onclick = resetFilters;
  

  document.getElementById('prev-page').onclick = () => {
    if (currentPagination > 1) {
      currentPagination--;
      loadtab();
    }
  };
  
  document.getElementById('next-page').onclick = () => {
    currentPagination++;
    loadtab();
  };



  
  document.getElementById('items-per-page').onchange = (e) => {
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
  
  
  initializeFilters();
  currentPagination = 1;



  loadtab();
}


function showMessage(message) {
  document.getElementById("vessels-tbody").innerHTML = 
    `<tr><td colspan="10">${message}</td></tr>`;
}






function applyFilters() { loadtab(); }
function refreshVisualization() { loadtab(); }
function predictClusters() { alert("FJe fais ca demain la team"); }
function predictSelected(type) { alert("Je fais ca demain la team, ou ce soir si j'ai le temps "); }
function exportData() { alert("Je fais ca demain la team"); }


