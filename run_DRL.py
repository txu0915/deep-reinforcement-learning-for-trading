# common library
import pandas as pd
import numpy as np
import time
import datetime as dt
#from stable_baselines3.common.vec_env import DummyVecEnv

from preprocessing.pull_and_process import *
from config import config
from model.models import *
import os


def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "done_data20220106.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20161001)].datadate.unique()
    print(len(unique_trade_date))

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    ## Ensemble Strategy
    run_ensemble_strategy(df=data,
                          unique_trade_date=unique_trade_date,
                          rebalance_window=rebalance_window,
                          validation_window=validation_window)

    # _logger.info(f"saving model version: {_version}")


if __name__ == "__main__":
    run_model()


# pd.read_csv('done_data20211227.csv').tic.unique()