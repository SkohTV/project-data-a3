import pandas as pd

from sklearn.pipeline import make_pipeline

# from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, GRU
from tensorflow import convert_to_tensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from data import create_train_test
from utils import *



@logged
def _evaluate_model(x_, y_, key: str, model, scaler, csv_size):
  log("Train/test split")
  x_train, x_test, y_train, y_test = train_test_split(x_, y_)
  y_test_2 = y_test

  log("Converting to tensors")
  x_train = convert_to_tensor(x_train, dtype=np.float64)
  x_test = convert_to_tensor(x_test, dtype=np.float64)
  y_train = convert_to_tensor(y_train, dtype=np.float64)
  y_test = convert_to_tensor(y_test, dtype=np.float64)

  filename = f'{key}_{csv_size}_v4.keras'
  print(filename)

  if not os.path.isfile(filename):
    log("Compile the model")
    model.compile(
      loss='mean_squared_error',
      optimizer='adam'
    )

    log("Fit the model")
    print(x_train)
    model.fit(
      x_train,
      y_train,
      epochs=100,
      batch_size=128,
      verbose=2
    ) 

    log("Exporting the model")
    model.save(filename)

  else:
    log("Loading the model")
    model = load_model(filename)

  log("Predicting")
  pred_test = model.predict(x_test)

  y_test = scaler.inverse_transform(y_test)
  pred_test = scaler.inverse_transform(pred_test)

  return (mean_absolute_error(y_test, pred_test, multioutput='raw_values')).tolist()



@logged
def test_neural(df: pd.DataFrame, csv_size: str, model: str):
  x_, y_, scaler = create_train_test(df)

  rnn = Sequential([
    Input((x_.shape[1],1)),
    LSTM(64), # Bigger if more input rows
    Dense(y_.shape[1]), # Output dimensions
  ])

  gru = Sequential([
    Input((x_.shape[1],1)),
    GRU(64), # Bigger if more input rows
    Dense(y_.shape[1]), # Output dimensions
  ])

  models = {
    "rnn": rnn,
    "gru": gru,
  }

  obj = _evaluate_model(x_, y_, model, models[model], scaler, csv_size)
  write_json(f'{model}_{csv_size}_eval.json', obj) # pyright: ignore

