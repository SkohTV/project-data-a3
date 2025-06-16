#!/usr/bin/env python3

import os
import time
import multiprocessing as mp

import pandas as pd

time_start = time.time()

def log(msg: str = ''):
  '''Print a log message with time and memory consumption'''
  global time_start
  time_spent = time.time() - time_start

  # https://stackoverflow.com/a/48397534
  with open('/proc/self/status') as f:
    memusage = int(f.read().split('VmRSS:')[1].split('\n')[0][:-3].strip()) / 1000

  print(f'[ {int(time_spent//60)}m{time_spent%60:.3f} ][ {memusage:.3f}mB ] {msg}')


def parse(f):
  r: pd.DataFrame = pd.read_csv(
    f,
    parse_dates=[1],
    engine='pyarrow', # SO IMPORTANT, BIG SPEED UP
  )
  log(f'Parsed {f}')

  r2 = r[
    (r["LON"] < -70) & (r["LON"] > -110) &
    (r["LAT"] < 35) & (r["LAT"] > 15) &
    (r["MMSI"] > 2715753) # Weird edgecase
  ]
  r2.loc[r2["Status"].isna(), "Status"] = 0
  r2.loc[r2["Heading"] == 511, "Heading"] = pd.NA
  log(f'Filtered {f}')

  return r2



def main():
  log("Starting, here be dragons!")

  ROOT_DIR = '/home/skoh/Downloads/share/'
  # ROOT_DIR = '/home/skoh/dev/repo/project-data-a3/big-data/sujet/'
  # ROOT_DIR = '/root/data_ais/csv'

  csv_files = [
    os.path.abspath(os.path.join(ROOT_DIR, f))
    for f in os.listdir(ROOT_DIR)
    if f.endswith('.csv')
  ]

  log("Parsing all csv files")
  with mp.Pool() as p:
    all_df = p.map(parse, csv_files)

  log("Merging dataframes")
  df: pd.DataFrame = pd.concat(all_df, ignore_index=True)
  print(len(pd.unique(df["MMSI"])))
  # print(df.isna().any())

  log("Sorting the dataframe")
  df.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)

  log("Exporting to csv")
  df.to_csv("output.csv", index=False)

  log("All done!")


if __name__ == '__main__':
  main()
