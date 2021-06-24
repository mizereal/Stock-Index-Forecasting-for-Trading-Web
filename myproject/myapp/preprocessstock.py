import numpy as np
import pandas as pd
import math, os, sys, datetime
from datetime import date, timedelta
import yfinance as yf
yf.pdr_override()
from functools import reduce
from ta.utils import dropna
from ta import add_all_ta_features
from sklearn import preprocessing
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

def preprocess(stock:str, timesteps:int=7):

    #Load data
    for i in range(727, 0, -1):
        startdate = date.today() - timedelta(i)
        enddate= date.today()
        ticker = yf.Ticker(stock) 
        data = ticker.history(start=startdate, end=enddate)
        if (len(data.index)%7 == 0):
            break  

    data.reset_index(inplace=True)
    
    # Add Indicator
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    # Create Label
    data['next_Close'] = data['Close'].shift(-7)
    data = data.drop(columns=['Volume', 'Open', 'High', 'Low','Dividends', 'Stock Splits'])
    
    #Get Date
    todate = data['Date'].shift(-7)
    todate = todate.dropna()
    
    # Feature Selection
    y = data['next_Close']
    y = y.dropna()
    featureScores = pd.DataFrame(data[data.columns[2:]].corr()['next_Close'][:])
    x_list = []
    for i in range(0, len(featureScores)):
        if abs(featureScores.next_Close[i]) > 0.80:
            x_list.append(featureScores.index[i])
    if len(x_list)<5:
        x_list.clear()
        for i in range(0, len(featureScores)):
            if abs(featureScores.next_Close[i]) > 0.05:
               x_list.append(featureScores.index[i])

    X = data[x_list]
    X = X.dropna()
    X = X.drop(columns=['next_Close'])
    sfs1 = SFS(LinearRegression(), k_features=(1,5), forward=True, floating=False, cv=0)
    sfs1.fit(X, y)
    k_feature_names = list(sfs1.k_feature_names_)
    features = data[k_feature_names]

    # Perporcess
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    features = features[:len(features)//timesteps*timesteps].reshape((len(features)//timesteps, timesteps, 5))
    
    labels = data[['next_Close']]
    getmax = labels.max()
    getmin = labels.min()
    labels = min_max_scaler.fit_transform(labels)
    labels = labels[:len(labels)//timesteps*timesteps].reshape((len(labels)//timesteps, timesteps, 1))
    labels = np.squeeze(labels)
    
    return todate, features, labels, getmax, getmin

def get_volumes(stock):
    for i in range(727, 0, -1):
        startdate = date.today() - timedelta(i)
        enddate= date.today()
        ticker = yf.Ticker(stock) 
        data = ticker.history(start=startdate, end=enddate)
        if (len(data.index)%7 == 0):
            break  

    ticker = yf.Ticker(stock) 
    data = ticker.history(start=startdate, end=enddate)
    data.reset_index(inplace=True)
    volumes = data['Volume'].to_numpy()

    return volumes
