
let map_visu = null;
let map_clusters = null;
let map_predict = null;

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

  for (let i in data) {

    let popup_msg = data[i][0]
    let color = data[i][1]
    let coords = data[i][2]

    map.on('load', () => {
      let id = String(uuidv4())

      // Add the line
      map.addSource(id, {
        'type': 'geojson',
        'data': generate_line(popup_msg, coords)
      });

      // Style the line
      map.addLayer({
        'id': id,
        'type': 'line',
        'source': id,
        'layout': { 'line-join': 'round', 'line-cap': 'round' },
        'paint': { 'line-color': color, 'line-width': 1 }
      });

      // Marker with popup
      const popup = new maplibregl.Popup({offset: 25}).setHTML(popup_msg);
      new maplibregl.Marker()
        .setOpacity(0.5)
        .setLngLat(coords[0])
        .setPopup(popup)
        .addTo(map);


      // https://maplibre.org/maplibre-gl-js/docs/examples/popup-on-click/
      // map.on('click', `id_${i}`, (e) => {
      //   new maplibregl.Popup()
      //     .setLngLat(e.lngLat)
      //     .setHTML(e.features[0].properties.description)
      //     .addTo(map);
      // });
      // map.on('mouseenter', `id_${i}`, () => map.getCanvas().style.cursor = 'pointer' );
      // map.on('mouseleave', `id_${i}`, () => map.getCanvas().style.cursor = '' );

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
        acc[mmsi] = { mmsi, vessel_name, length, width, color: '#F00', vals: [] };
      acc[mmsi].vals.push([longitude, latitude]);
      last = {latitude: latitude, longitude: longitude, sog: sog, cog: cog, heading: heading, vessel_name: vessel_name}
      return acc;
    }, {});

    const val_array = Object.values(tr);
    const pre_c = val_array.map(x => Object.values(x));
    const c1 = pre_c.map(x => [generate_popup_txt(x[0], x[1], x[2], x[3]), x[4], x[5]]);

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
      let popup_txt = generate_popup_txt(mmsi, vessel_name, length, width)
      let c = [[popup_txt, '#0F0', predicted_trajectory]]

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
