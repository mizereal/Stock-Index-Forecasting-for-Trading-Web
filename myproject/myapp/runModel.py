from .predictor import Predictor
from .pathModel import pathModel
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from datetime import date, timedelta
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error  as MSE

def runModel(model_n, stock, type_y:bool=True):
    model = Predictor(model_n, 32)
    todate, x, y, max, min = model.load_transform(stock)

    pred = model.predict(x)

    pred = pred.reshape(1,-1)[0]
    for i in range(0, len(pred)):
        pred[i] = (pred[i]*(max-min))+min

    y = y.reshape(1,-1)[0]
    for i in range(0, len(y)):
        y[i] = (y[i]*(max-min))+min 
    
    y = y[~np.isnan(y)]

    pred_test = pred[0:-7]

    RMSE =round(math.sqrt(MSE(y, pred_test)), 4)
    EVS = round(explained_variance_score(y, pred_test), 4)

    model_n = pathModel(model_n)
    df = pd.DataFrame({"Model":[model_n], "RMSE":[RMSE] ,"EVS":[EVS]})

    actual = y[-28:]
    trend = pred[-35:-6]
    predict = pred[-7:]
    
    a = date.today()
    todate = todate.apply(lambda x: x.strftime('%d%b'))
    todate = todate.values.tolist()
    
    if type_y==True:
        dateList = []
        for i in range (0, 7):
           d = a + timedelta(days = i)
           d = d.strftime("%d%b")
           dateList.append(d) 
        todate_p = todate[-28:]
        todate_p.append(dateList[0]) 
        todate = todate[-28:]
    else:
        actual = np.diff(actual) / actual[1:] * 100
        trend = np.diff(trend) / trend[1:] * 100
        predict = np.diff(predict) / predict[1:] * 100
        trend = trend[-27:]
        trend = np.append(trend, predict[0])
        dateList = []
        for i in range (0, 6):
           d = a + timedelta(days = i)
           d = d.strftime("%d%b")
           dateList.append(d)
        todate_p = todate[-27:]
        todate_p.append(dateList[0])
        todate = todate[-27:]


    return actual, trend, predict, dateList, todate, todate_p, df