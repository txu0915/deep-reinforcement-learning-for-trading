# common library
import pandas as pd
import numpy as np
import time
import datetime as dt
from stable_baselines3.common.vec_env import DummyVecEnv

from preprocessing.pull_and_process import *
from config import config
from model.models import *
import os

def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    today = int(dt.datetime.now().strftime('%Y%m%d'))
    tic_list = config.model_tic_list
    preprocessed_path = f"done_data{str(today)}.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
        data = data[data.tic.isin(tic_list)]
    else:
        data = preprocess_data()
        data = data[data.tic.isin(tic_list)]
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)
    print(data.dtypes)


    start = int((dt.datetime.now() + dt.timedelta(days=-365)).strftime('%Y%m%d'))
    unique_trade_date = data[(data.datadate > start)&(data.datadate <= today)].datadate.unique()
    

    # rebalance_window is the number of days to retrain the model
    # validation_window is the number of days to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63
    unique_trade_date = sorted(unique_trade_date)[-(validation_window+2):]
    print(unique_trade_date)
    ## Ensemble Strategy
    run_ensemble_strategy(df=data, 
                          unique_trade_date= unique_trade_date,
                          rebalance_window = rebalance_window,
                          validation_window=validation_window)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
