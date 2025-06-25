
let map_visu = null;
let map_clusters = null;
let map_predict = null;


// id either vessels-map or clusters-map or predict-map
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
        center: [-80, 20],
        zoom: 3
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
                'paint': { 'line-color': color, 'line-width': 2 }
            });

            // https://maplibre.org/maplibre-gl-js/docs/examples/popup-on-click/
            map.on('click', `id_${i}`, (e) => {
                new maplibregl.Popup()
                    .setLngLat(e.lngLat)
                    .setHTML(e.features[0].properties.description)
                    .addTo(map);
            });
            map.on('mouseenter', `id_${i}`, () => map.getCanvas().style.cursor = 'pointer' );
            map.on('mouseleave', `id_${i}`, () => map.getCanvas().style.cursor = '' );

        });

    }
}





c = [
    ['hello1', '#F00', [[-100, 0], [-80, 20]]],
    ['hello2', '#0F0', [[-40, 39], [-32, 10]]],
    ['hello3', '#00F', [[-69, 9], [-12, 0]]],
]

map_visu = generate_map('visu')
map_clusters = generate_map('clusters')
map_predict = generate_map('predict')
add_lines(map_visu, c)

function predict_trajectoire_vesseltype(row) {
    DEFAULT_PREDICT_VESSEL_TYPE = 80;

    const params = new URLSearchParams({
        latitude: row[0],
        longitude: row[1],
        sog: row[2],
        cog: row[3],
        heading: row[4],
        vesseltype: DEFAULT_PREDICT_VESSEL_TYPE,
        steps: 15,
    })

    ajaxRequest("GET", `php/requests.php/predict_boat_trajectory?${params}`, (r) => {
       console.log(r) 
    });

}

DEFAULT_PREDICT = [24.522600, -83.732800, 12.2, 113.1, 115];

predict_trajectoire_vesseltype(DEFAULT_PREDICT)
