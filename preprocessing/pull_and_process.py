#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:52:51 2021

@author: qinfang
"""
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config.config import *
# import pandas_market_calendars as mcal
import datetime as dt
import yahoo_fin.stock_info as si
import yfinance as yf
import os.path



def load_dataset(tic_list):
    end = dt.datetime.now()
    lookback_days = LOOKBACK_DAYS
    stock_file = TRAINING_DATA_FILE
    stocks = []
    for tic in tic_list:
        print("....",tic, '....')
        df_tic = yf.download(tic,  end + dt.timedelta(days=-lookback_days), end, progress=False)
        df_tic = df_tic.reset_index()
        df_tic['tic'] = tic
        stocks.append(df_tic)
    if len(stocks) > 0:
        df = pd.concat(stocks)
        df = df.drop_duplicates(subset=['tic','Date'])
        df.to_csv(stock_file, index=False)
        return df
    return None

def pull_data_by_tic(ticker, start, end):
    df_ticker = yf.download(ticker,start,end, progress=False)
    df_ticker = df_ticker.reset_index()
    df_ticker['tic'] = ticker
    return df_ticker
    


def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date > start) & (df.date <= end)]
    data=data.sort_values(['date','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.date.factorize()[0]
    return data

def rename_columns(df):
    data = df.copy()
    data = data[['Date','tic', 'Adj Close', 'Open', 'High', 'Low', 'Volume']]
    data = data.rename(columns={'Date':'datadate','Adj Close':'adjcp','Open':'open','High':'high','Low':'low','Volume':'volume'})
    data['datadate'] = pd.to_datetime(data['datadate'])
    data['datadate'] = data['datadate'].apply(lambda x: int(dt.datetime.strftime(x,'%Y%m%d')))
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data



def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()


    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)


    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df



def preprocess_data(tic_list):
    """data preprocessing pipeline"""
    df = load_dataset(tic_list)
    df = rename_columns(df)
    df = add_technical_indicator(df)
    df.fillna(method='bfill',inplace=True)

    return df

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets
    
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    
    
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index












