from data import *
from regr import *
from rnn import *


SMALL_CSV = './data/small.csv'
LARGE_CSV = './data/large.csv'


if __name__ == '__main__':

  df = parse_csv(LARGE_CSV)
  add_columns(df, remove_stale=True)
  #
  # log("RNN models")
  # for md in ['rnn', 'gru']:
  #   test_neural(df, model=md, csv_size='small')

  # log("Regression models")
  # for md in ['linear', 'randomforest', 'polynominal', 'svr', 'histgradient']:
  #   test_regressions(df, model=md, csv_size='small', reps=1)
  #
  #
  # df = parse_csv(LARGE_CSV)
  # add_columns(df, remove_stale=True)
  #
  # log("Regression models")
  # for md in ['randomforest', 'histgradient']:
  #   test_regressions(df, model=md, csv_size='large', reps=1)
  #
  # log("RNN models")
  # for md in ['rnn', 'gru']:
  #   test_neural(df, model=md, csv_size='large')
  #
  # log("Regression models")
  # for md in ['polynominal', 'svr']:
  #   test_regressions(df, model=md, csv_size='large', reps=1)
  #
