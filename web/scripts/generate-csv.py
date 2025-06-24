#!/usr/bin/env python3

import csv
from io import StringIO
import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore")




def prediction(lat, lon, sog, cog, heading):
    pipeline = joblib.load('../../ia/data/model_1.pkl')
    model = pipeline['model']
    scaler = pipeline['scaler']

    data = pd.DataFrame([[lat, lon, sog, cog, heading]], 
                       columns=['LAT', 'LON', 'SOG', 'COG', 'Heading'])
    
    # on normalise et on prédit
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)[0]

    return cluster






if __name__ == "__main__":

    headers = ["MMSI","BaseDateTime","LAT","LON","SOG","COG","Heading","VesselName","IMO","CallSign","VesselType","Status","Length","Width","Draft","Cargo","TransceiverClass"]
    out = StringIO()
    out.write(','.join(headers) + ',Cluster\n')
    j = 0

    with open('../../ia/data/large.csv', 'r') as f:

        rdr = csv.reader(f)
        rdr.__next__()

        for i in rdr:

            if not j % 1000:
                print(f'{j//1000}:_/5_302')
            j+=1

            row = {k: v for k, v in zip(headers, i)}
            pred = prediction(row['LAT'], row['LON'], row['SOG'], row['COG'], row['Heading'])
            out.write(','.join(i) + ',' + str(pred) + '\n')

    with open('./output.csv', 'w') as f:
        f.write(out.getvalue())

