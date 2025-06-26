
let map_visu = null;
let map_clusters = null;
let map_predict = null;

// https://stackoverflow.com/a/2117523
function uuidv4() {
  return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
    (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
  );
}

// id either visu, clusters or predict
function generate_map(id) {
  if (id == 'visu')
    container = 'vessels-map'
  if (id == 'clusters')
    container = 'clusters-map'
  if (id == 'predict')
    container = 'predict-map'

  return new maplibregl.Map({
    container: container,
    style: 'https://demotiles.maplibre.org/style.json',
    center: [-88, 25],
    zoom: 4
  });
}



function generate_line(popup_msg, coords) {
  return {
    'type': 'Feature',
    'properties': { 'description': popup_msg },
    'geometry': { 'type': 'LineString', 'coordinates': coords }
  }
}


// Data format:
// [
//   [popup_msg, color, [[LAT1, LON1], [LAT2, LON2], ...]],
//   [popup_msg, color, [[LAT1, LON1], [LAT2, LON2], ...]],
//   [popup_msg, color, [[LAT1, LON1], [LAT2, LON2], ...]],
// ]
function add_lines(map, data) {

  let all = {}

  for (let i in data) {

    let popup_msg = data[i][0]
    let color = data[i][1]
    let coords = data[i][2]
    let id = String(uuidv4())

    if (! (color in all)) all[color] = {}
    all[color][id] = generate_line(popup_msg, coords)
  }


  for (let a in all) {
    let id = String(uuidv4())

    map.on('load', () => {

      // Add the line
      map.addSource(id, {
        'type': 'geojson',
        'data': {
          'type': 'FeatureCollection',
          'features': Object.values(all[a])
        }
      });

      // Style the line
      map.addLayer({
        'id': id,
        'type': 'line',
        'source': id,
        'layout': { 'line-join': 'round', 'line-cap': 'round' },
        'paint': { 'line-color': a, 'line-width': 2 }
      });

      // https://maplibre.org/maplibre-gl-js/docs/examples/popup-on-hover/
      const popup = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false
      });

      let currentFeatureCoordinates = undefined;
      map.on('mousemove', id, (e) => {
          const featureCoordinates = e.features[0].geometry.coordinates.toString();
          if (currentFeatureCoordinates !== featureCoordinates) {
              currentFeatureCoordinates = featureCoordinates;

              map.getCanvas().style.cursor = 'pointer';
              const description = e.features[0].properties.description;

              popup
                .setLngLat(e.lngLat)
                .setHTML(description)
                .addTo(map);
          }
      });

      map.on('mouseleave', id, () => {
          currentFeatureCoordinates = undefined;
          map.getCanvas().style.cursor = '';
          popup.remove();
      });

    });

  }
}


function predict_trajectoire_vesseltype() {
  DEFAULT_PREDICT_VESSEL_TYPE = 80;
  let mmsi = document.querySelector('#vessels-table input[type="radio"]:checked').dataset['mmsi']

  const params = new URLSearchParams({
    longueur_min: document.getElementById("filter-longueur-min").value,
    longueur_max: document.getElementById("filter-longueur-max").value,
    largeur_min: document.getElementById("filter-largeur-min").value,
    largeur_max: document.getElementById("filter-largeur-max").value,
    mmsi: mmsi
  });

  ajaxRequest('GET', `php/requests.php/all_points_donnee?${params}`, (r) => {
    let last = {}

    const tr = r.reduce((acc, { mmsi, vessel_name, length, width, latitude, longitude, sog, cog, heading }) => {
      if (!acc[mmsi])
        acc[mmsi] = { mmsi, vessel_name, length, width, color: '#0A0', vals: [] };
      acc[mmsi].vals.push([longitude, latitude]);
      last = {latitude: latitude, longitude: longitude, sog: sog, cog: cog, heading: heading, vessel_name: vessel_name, length: length, width: width}
      return acc;
    }, {});

    const val_array = Object.values(tr);
    const pre_c = val_array.map(x => Object.values(x));
    const c1 = pre_c.map(x => [generate_popup_txt(x[0], x[1], last['length'], last['width']), x[4], x[5]]);
    console.log(val_array)

    const params = new URLSearchParams({
      latitude: last['latitude'],
      longitude: last['longitude'],
      sog: last['sog'],
      cog: last['cog'],
      heading: last['heading'],
      vesseltype: DEFAULT_PREDICT_VESSEL_TYPE,
      steps: 1000,
    })

    let predicted_trajectory = null;
    let vessel_name = last['vessel_name']

    ajaxRequest("GET", `php/requests.php/predict_boat_trajectory?${params}`, (r) => {
      predicted_trajectory = r.map((x) => [JSON.parse(x).LON[0], JSON.parse(x).LAT[0]])
      // ajaxRequest("GET", `php/requests.php/fetch_boat_picture?mmsi=${mmsi}`, (r) => {
      let popup_txt = generate_popup_txt(mmsi, vessel_name, last['length'], last['width'])
      let c = [[popup_txt, '#C00', predicted_trajectory]]

      map_predict = generate_map('predict')
      add_lines(map_predict, c1)
      add_lines(map_predict, c)
      // })
    });
  })


}

function generate_popup_txt(mmsi, name, length, width) {
  return `<div>
    <h1>${name}</h1>
    <p>${mmsi}</p>
    <br>
    <p>${length}m X ${width}m</p>
  </div>`
}


function predict_clusters() {
  colors = ['#F00', '#0F0', '#00F', '#FF0', '#F0F', '#0FF', '#FFF']


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


  ajaxRequest('GET', `php/requests.php/all_points_donnee?${params}`, (r) => {

    const tr = r.reduce((acc, { mmsi, vessel_name, length, width, latitude, longitude, id_cluster }) => {

      if (!acc[mmsi])
        acc[mmsi] = { mmsi, vessel_name, length, width, color: colors[Number(id_cluster)-1], vals: [] };

      acc[mmsi].vals.push([longitude, latitude]);
      return acc;

    }, {});

    const val_array = Object.values(tr);
    const pre_c = val_array.map(x => Object.values(x));
    const c = pre_c.map(x => [generate_popup_txt(x[0], x[1], x[2], x[3]), x[4], x[5]]);

    map_clusters = generate_map('clusters')
    add_lines(map_clusters, c)
  })
}





      // https://maplibre.org/maplibre-gl-js/docs/examples/popup-on-click/
      // map.on('click', `id_${i}`, (e) => {
      //   new maplibregl.Popup()
      //     .setLngLat(e.lngLat)
      //     .setHTML(e.features[0].properties.description)
      //     .addTo(map);
      // });
      // map.on('mouseenter', `id_${i}`, () => map.getCanvas().style.cursor = 'pointer' );
      // map.on('mouseleave', `id_${i}`, () => map.getCanvas().style.cursor = '' );
