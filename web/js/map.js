


let map_visu = new maplibregl.Map({
    container: 'vessels-map',
    style: 'https://demotiles.maplibre.org/style.json',
    center: [-80, 20],
    zoom: 3
});



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

add_lines(map_visu, c)
