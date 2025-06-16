import os
import json
import time
from functools import wraps

import numpy as np


__log_stack = list()
__start_time = time.time()
STORAGE = './output/'

def memusage():  # https://stackoverflow.com/a/48397534
  with open('/proc/self/status') as f:
    return int(f.read().split('VmRSS:')[1].split('\n')[0][:-3].strip()) / 1_000_000

def get_time():
  return time.time() - __start_time

def log(msg: str):
  tm = get_time()
  print(
    f'[ {int(tm // 60)}m{tm%60:.1f}s ]',
    f'[ ~{memusage():.3f}Gb ]',
    *[f'({i})' for i in __log_stack],
    ' ',
    msg,
    sep=''
  )

def _push_log(scope: str): __log_stack.append(scope)
def _pop_log(): __log_stack.pop()


def logged(fn):

  @wraps(fn)
  def wrapped(*args, **kwargs):
    _push_log(fn.__name__)
    ret = fn(*args, **kwargs)
    _pop_log()
    return ret

  return wrapped


@logged
def write_json(fname: str, data):
  log("Exporting to json")
  with open(os.path.join(STORAGE, fname), 'w') as f:
    json.dump(data, f)


# https://www.geeksforgeeks.org/dsa/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
def haversine(lat1, lon1, lat2, lon2):
    
    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1) * np.pi / 180.0
    dLon = (lon2 - lon1) * np.pi / 180.0

    # convert to radians
    lat1 = (lat1) * np.pi / 180.0
    lat2 = (lat2) * np.pi / 180.0

    # apply formulae
    a = (pow(np.sin(dLat / 2), 2) + 
         pow(np.sin(dLon / 2), 2) * 
             np.cos(lat1) * np.cos(lat2));
    rad = 6371
    c = 2 * np.asin(np.sqrt(a))
    return rad * c * 1000



