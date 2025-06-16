from multiprocessing import Pool
from functools import partial

import pandas as pd
import numpy as np
import sklearn as sk

from pickle import dump, load
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

from data import create_train_test
from utils import *



@logged
def _evaluate_model(x_, y_, model, model_name, scaler):
    log("Train/test split")
    x_train, x_test, y_train, y_test = train_test_split(x_, y_)

    if not os.path.isfile(f"{model_name}_v4.pkl"):
      log("Fitting model")
      reg = model.fit(x_train, y_train)
    else:
      log("Loading model from file")
      with open(f"{model_name}_v3.pkl", "rb") as f:
        reg = load(f)

    log("Predicting")
    pred = reg.predict(x_test)

    if not os.path.isfile(f"{model_name}_v3.pkl"):
      log("Exporting model")
      with open(f"{model_name}_v3.pkl", "wb") as f:
        dump(reg, f, protocol=5)

    y_test = scaler.inverse_transform(y_test)
    pred = scaler.inverse_transform(pred)

    return mean_absolute_error(y_test, pred, multioutput='raw_values')


@logged
def _full_evaluate_model(x_, y_, key: str, model, reps: int, scaler, multithread=False):
  log(f'Creating pool of model {key} (multithread={multithread})')
  if multithread:
    with Pool() as p:
      dst = p.starmap(partial(_evaluate_model, x_=x_, y_=y_, model=model, model_name=key, scaler=scaler), [() for _ in range(reps)])
  else:
      dst = [_evaluate_model(x_, y_, model, key, scaler) for _ in range(reps)]

  dst = [i.tolist() for i in dst]

  # dst = [haversine(i, i, 0, 0) for i in val]
  return {'name': key, 'mean': np.mean(dst), 'data': dst}


@logged
def test_regressions(df: pd.DataFrame, csv_size: str, model: str, reps: int):
  x_, y_, scaler = create_train_test(df)

  models = {
    "linear" : make_pipeline(StandardScaler(), LinearRegression()),
    "polynominal" : make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()),
    "svr" : make_pipeline(StandardScaler(), MultiOutputRegressor(LinearSVR(max_iter=10_000))),
    "randomforest" : make_pipeline(StandardScaler(), RandomForestRegressor()),
    "histgradient" : make_pipeline(StandardScaler(), MultiOutputRegressor(HistGradientBoostingRegressor())),
  }

  log("Creating pool of all models")
  obj = _full_evaluate_model(x_, y_, model, models[model], reps=reps, multithread=False, scaler=scaler)
  write_json(f'{model}_{csv_size}_eval.json', obj) # pyright: ignore

