
document.getElementById("filter-mmsi").oninput = () => {
  const mmsi = document.getElementById("filter-mmsi").value.trim();
  if (mmsi.length >= 3) {
    
    
    ajaxRequest('GET', `php/requests.php/all_mmsi?mmsi=${mmsi}`, (response) => {
        const dataList = document.getElementById("filter-mmsi-list");
        dataList.innerHTML = "";
        response.forEach((item) => {
            const option = document.createElement("option");
            option.value = item.mmsi;
            dataList.appendChild(option);
        });
        }
    );



  }
}
