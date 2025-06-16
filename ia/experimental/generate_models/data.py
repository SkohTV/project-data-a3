from pickle import dump
from typing import Tuple
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from utils import *



@logged
def parse_csv(path: str):
  log("Reading csv")
  df = pd.read_csv(path, parse_dates=[1])

  log("Sorting values")
  df.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)
  df.reset_index(drop=True, inplace=True)

  return df


@logged
def add_columns(df: pd.DataFrame, remove_stale = True):

  if remove_stale:
    log("Remove stale boats")
    mask = (df['SOG'] < 0.3)
    df.drop(index=df.index[mask], inplace=True) # pyright: ignore

  log("Shifting")
  df["NextLAT"] = df["LAT"].shift(-1) - df["LAT"] # delta
  df["NextLON"] = df["LON"].shift(-1) - df["LON"] # delta
  df["NextSOG"] = df["SOG"].shift(-1) - df["SOG"]
  df["NextTime"] = df["BaseDateTime"].shift(-1) - df["BaseDateTime"]

  log("Change COG/Heading for +sin/+cos")
  df["SinCOG"] = df["COG"].apply(np.sin)
  df["CosCOG"] = df["COG"].apply(np.cos)
  df.drop(columns="COG", inplace=True)

  df["SinHeading"] = df["Heading"].apply(np.sin)
  df["CosHeading"] = df["Heading"].apply(np.cos)
  df.drop(columns="Heading", inplace=True)

  log("Shifting for cos/sin")
  df["NextSinCOG"] = df["SinCOG"].shift(-1) - df["SinCOG"]
  df["NextCosCOG"] = df["CosCOG"].shift(-1) - df["CosCOG"]
  df["NextSinHeading"] = df["SinHeading"].shift(-1) - df["SinHeading"]
  df["NextCosHeading"] = df["CosHeading"].shift(-1) - df["CosHeading"]

  log("Type casts")
  df["NextTime"] = df["NextTime"].apply(pd.Timedelta.total_seconds)
  df["VesselType"] = df["VesselType"].astype(str)

  log("Dropping wrong dupes")
  mask = (df['MMSI'] != df['MMSI'].shift(-1))
  df.drop(index=df.index[mask], inplace=True) # pyright: ignore
  df.reset_index(drop=True, inplace=True)
  print(df)


@logged
def create_train_test(df: pd.DataFrame):
  log("Scaling features with MinMaxScaler")
  scalerX = MinMaxScaler(feature_range=(0, 1))
  scalerY = MinMaxScaler(feature_range=(0, 1))

  log("Creating X/Y")
  x_ = pd.get_dummies(df[["LAT", "LON", "SOG", "SinCOG", "CosCOG", "SinHeading", "CosHeading", "VesselType", "NextTime"]])
  y_ = df[["NextLAT", "NextLON", "NextSOG", "NextSinCOG", "NextCosCOG", "NextSinHeading", "NextCosHeading"]]
  with open(f"empty_df.pkl", "wb") as f:
    dump(x_.iloc[0:0], f, protocol=5)

  log("Scale features")
  x_ = scalerX.fit_transform(x_)
  y_ = scalerY.fit_transform(y_)

  with open(f"scalerX.pkl", "wb") as f:
    dump(scalerX, f, protocol=5)
  with open(f"scalerY.pkl", "wb") as f:
    dump(scalerY, f, protocol=5)

  return (x_, y_, scalerY) # pyright: ignore
