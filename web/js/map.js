
let map_visu = null;
let map_clusters = null;
let map_predict = null;


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

      // Add the line
      map.addSource(`id_${i}`, {
        'type': 'geojson',
        'data': generate_line(popup_msg, coords)
      });

      // Style the line
      map.addLayer({
        'id': `id_${i}`,
        'type': 'line',
        'source': `id_${i}`,
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


function predict_trajectoire_vesseltype(row) {
  DEFAULT_PREDICT_VESSEL_TYPE = 80;
  let mmsi = document.querySelector('#vessels-table input[type="radio"]:checked').dataset['mmsi']
  let name = 'unnamed'

  const params = new URLSearchParams({
    latitude: row[2],
    longitude: row[3],
    sog: row[4],
    cog: row[5],
    heading: row[6],
    vesseltype: DEFAULT_PREDICT_VESSEL_TYPE,
    steps: 1000,
  })

  let predicted_trajectory = null;

  ajaxRequest("GET", `php/requests.php/predict_boat_trajectory?${params}`, (r) => {
    predicted_trajectory = r.map((x) => [JSON.parse(x).LON[0], JSON.parse(x).LAT[0]])


    ajaxRequest("GET", `php/requests.php/fetch_boat_picture?mmsi=${mmsi}`, (r) => {
      let popup_txt = `<div><h1>${name}</h1><image class='boat-pic' src=${r}></div>`
      let c = [[popup_txt, '#F00', predicted_trajectory]]
      map_predict = generate_map('predict')
      add_lines(map_predict, c)
    })
  });
}

function generate_popup_txt(mmsi, name, length, width) {
  return `<div>
    <h1>${name}</h1>
    <p>${mmsi}</p>
    <br>
    <p>${length}m X ${width}m</p>
  </div>`
}
