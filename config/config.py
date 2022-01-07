import pathlib

#import finrl

import pandas as pd
import datetime
import os
import yahoo_fin.stock_info as si



TRAINING_DATA_FILE = "data/stock.csv"
tic_list = si.tickers_dow()
model_tic_list = ['AAPL','AMGN','AXP','BA','CAT','CRM','CSCO','CVX','DIS', 'GS','HD','HON','IBM','INTC',
                  'JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','PG','TRV','UNH','V','VZ','WBA','WMT']
## missing 'DOW'
# tick_file_dow30 = '/Users/tianlongxu/financial-projects/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/data/dow_30_2009_2020.csv'
# tics = pd.read_csv(tick_file_dow30)
# model_tic_list = tics.tic.unique()
# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000
# total number of stocks in our portfolio
STOCK_DIM = len(model_tic_list)
state_dim = 1 + STOCK_DIM*6
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

lookback_days = 3000
# now = datetime.datetime.now().strftime('%Y%m%d%H%M')
now = datetime.datetime.now().strftime('%Y%m%d')
TRAINED_MODEL_DIR = f"trained_models/{now}"
#TRAINED_MODEL_DIR = "trained_models/model"
if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"



