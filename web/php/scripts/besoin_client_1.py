#!/usr/bin/env python3

import joblib
import argparse
import pandas as pd



def get_cluster_indication(cluster_id):
    
    indications = {
        0: "Est probablement dans le cluster des navires qui sont proches des cotes de Floride",
        1: "Est probablement dans un port dans la zone Houston/Nouvelle-Orléans", 
        2: "Est probablement en transit dans la zone centrale du Golfe du Mexique",
        3: "Est probablement dans un corridor commerciale majeur de la côte Est",
        4: "Est probablmeent en déplacement commercial rapide vers l'Amérique Centrale",
        5: "Est probablement sur une route commerciale vers l'ouest/Mexique",
        6: "Est probablement a l'arret proche des cotes dans le golfe du Mexique"
    }
    return indications.get(cluster_id, "Comportement pas trouvé")




def prediction(lat, lon, sog, cog, heading):
    # (modèle + scaler)
    pipeline = joblib.load('models/model_1.pkl')
    model = pipeline['model']
    scaler = pipeline['scaler']

    data = pd.DataFrame([[lat, lon, sog, cog, heading]], 
                       columns=['LAT', 'LON', 'SOG', 'COG', 'Heading'])
    
    # on normalise et on prédit
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)[0]

    print(f"Ce navire est dans le cluster: {cluster}")

    print(get_cluster_indication(cluster))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering maritime")
    parser.add_argument('--lat', type=float, required=True, help='Latitude du navire (14-35)')
    parser.add_argument('--lon', type=float, required=True, help='Longitude du navire (-110 à 75)')
    parser.add_argument('--sog', type=float, required=True, help='Speed Over Ground (0-27 nœuds)')
    parser.add_argument('--cog', type=float, required=True, help='Course Over Ground (0-360°)')
    parser.add_argument('--heading', type=float, required=True, help='Heading (0-360°)')
    
    args = parser.parse_args()

    prediction(args.lat, args.lon, args.sog, args.cog, args.heading)

