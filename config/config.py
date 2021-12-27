import pathlib

#import finrl

import pandas as pd
import datetime
import os
import yahoo_fin.stock_info as si

TRAINING_DATA_FILE = "data/stock.csv"
tic_list = si.tickers_dow()
model_tic_list = ['AAPL',
 'AMGN',
 'AXP',
 'BA',
 'CAT',
 'CRM',
 'CSCO',
 'CVX',
 'DIS',
 'GS',
 'HD',
 'HON',
 'IBM',
 'INTC',
 'JNJ',
 'JPM',
 'KO',
 'MCD',
 'MMM',
 'MRK',
 'MSFT',
 'NKE',
 'PG',
 'TRV',
 'UNH',
 'V',
 'VZ',
 'WBA',
 'WMT']

lookback_days = 3000
now = datetime.datetime.now().strftime('%Y%m%d%H%M')
TRAINED_MODEL_DIR = f"trained_models/{now}"
#TRAINED_MODEL_DIR = "trained_models/model"
if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"


