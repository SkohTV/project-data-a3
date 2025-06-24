#!/usr/bin/env python

import csv
from io import StringIO

vessels = StringIO()
points = StringIO()
boats = set()

headers = ["MMSI","BaseDateTime","LAT","LON","SOG","COG","Heading","VesselName","IMO","CallSign","VesselType","Status","Length","Width","Draft","Cargo","TransceiverClass","Cluster"]
# vessels.write('INSERT INTO vessel (mmsi,vessel_name,imo_number,callsign,length,width,code_transceiver) VALUES \n')
intro = 'INSERT INTO point_donnee (base_date_time,mmsi,latitude,longitude,speed_over_ground,cap_over_ground,heading,draft,code_status,id_cluster) VALUES \n'
j = 0


def transceiver(x):
    return '1' if x == 'A' else '2'

def escape(x):
    return f'"{x.replace('"', '\\"')}"'


with open('./output.csv', 'r') as f:
    rdr = csv.reader(f)
    rdr.__next__()

    for itm in rdr:

        if not j % 1000:
            print(f'{j//1000:_}/5_302')
        j+=1

        row = {k: v for k, v in zip(headers, itm)}

        # if row['MMSI'] not in boats:
        #     boats.add(row['MMSI'])
        #     vessels.write('(' + ','.join([row['MMSI'], escape(row['VesselName']), escape(row['IMO']), escape(row['CallSign']), row['Length'], row['Width'], transceiver(row['TransceiverClass'])]) + '),\n')

        points.write('(' + ','.join([escape(row['BaseDateTime']), row['MMSI'], row['LAT'], row['LON'], row['SOG'], row['COG'], row['Heading'], row['Draft'], row['Status'], str(int(row['Cluster'])+1)]) + '),\n')


slices = points.getvalue()[:-1].split('\n')
ret = [slices[i:i+10000] for i in range(len(slices) // 10000 + 1)]

for idx, itm in enumerate(ret):
    with open(f'points_final_{idx}.sql', 'w') as f:
        f.write(intro + '\n'.join(itm)[:-1] + ';')


