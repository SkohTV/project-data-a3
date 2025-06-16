#!/usr/bin/env python3

import pickle
import argparse
import json
import os

import numpy as np
import pandas as pd


DATA_FILES = os.path.realpath("../data/")


# ----------------------------------------------


# Parser setup
parser = argparse.ArgumentParser(
  prog='Besoin client 1',
  description='Prédit la future position/trajectoire d\'un navire en fonction de sa position actuelle (+ d\'autres informations)',
)

parser.add_argument('--steps', help='number of next 5min steps to compute', type=int, default=1)
parser.add_argument('--json', help='{"LAT": ..., "LON": ..., "SOG": ..., "COG": ..., "Heading": ..., "VesselType": ...}')

args = parser.parse_args()


# ----------------------------------------------


# Parse the node
if not args.json:
  node = {
    'LAT': float(input("LAT: ")),
    'LON': float(input("LON: ")),
    'SOG': float(input("SOG: ")),
    'COG': float(input("COG: ")),
    'Heading': float(input("Heading: ")),
    'VesselType': int(input("VesselType: ")),
  }
else:
  node = json.loads(args.json)

assert isinstance(node['LAT'], float), "'LAT' n'est pas un float"
assert isinstance(node['LON'], float), "'LON' n'est pas un float"
assert isinstance(node['SOG'], float), "'SOG' n'est pas un float"
assert isinstance(node['COG'], float), "'COG' n'est pas un float"
assert isinstance(node['Heading'], float), "'Heading' n'est pas un float"
assert isinstance(node['VesselType'], int), "'VesselType' n'est pas un int"


# ----------------------------------------------


# Load scalers & empty df
with open(os.path.join(DATA_FILES, f'empty_df.pkl'), 'rb') as f:
  data = pickle.load(f)

with open(os.path.join(DATA_FILES, f'scalerX.pkl'), 'rb') as f:
  scalerX = pickle.load(f)
with open(os.path.join(DATA_FILES, f'scalerY.pkl'), 'rb') as f:
  scalerY = pickle.load(f)


def dict_to_predictable(dct: dict):
  '''Convert a dict to a predictable ndarray'''
  inpt = pd.get_dummies(pd.DataFrame(dct))
  inpt = inpt.reindex(columns=data.columns, fill_value=False)
  inpt = scalerX.transform(inpt)
  return inpt


# First input
current = {
  'LAT': [node['LAT']],
  'LON': [node['LON']],
  'SOG': [node['SOG']],
  'SinCOG': [np.sin(node['COG'])],
  'CosCOG': [np.cos(node['COG'])],
  'SinHeading': [np.sin(node['Heading'])],
  'CosHeading': [np.cos(node['Heading'])],
  'VesselType': [str(node['VesselType'])],
  'NextTime': [300] # next 5 mins
}

current_to_pred = dict_to_predictable(current)


# ----------------------------------------------



with open(os.path.join(DATA_FILES, 'linear_v3.pkl'), 'rb') as f:
  model = pickle.load(f)


for i in range(args.steps):
  pred = model.predict(current_to_pred)
  pred = scalerY.inverse_transform(pred)[0]

  print(f'{{"LAT": {current['LAT'] + pred[0]}, "LON": {current['LON'] + pred[0]}}}')

  current = {
    'LAT': current['LAT'] + pred[0],
    'LON': current['LON'] + pred[1],
    'SOG': current['SOG'] + pred[2],
    'SinCOG': current['SinCOG'] + pred[3],
    'CosCOG': current['CosCOG'] + pred[4],
    'SinHeading': current['SinHeading'] + pred[5],
    'CosHeading': current['SinHeading'] + pred[6],
    'VesselType': current['VesselType'],
    'NextTime': [300] # next 5 mins
  }
  current_to_pred = dict_to_predictable(current)

