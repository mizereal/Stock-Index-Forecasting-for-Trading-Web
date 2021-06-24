from myapp.preprocessstock import preprocess
from django.shortcuts import render
from django.http import HttpResponse
from .predictor import Predictor
from .preprocessstock import preprocess
from .plot import get_plot
from .pathTA import pathTA
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from datetime import date, timedelta
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error  as MSE

def gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
def home(request):
    return render(request, "home.html", {})

def set100(request):
    return render(request, "set100.html", {})

def djia(request):
    return render(request, "djia.html", {})

def djiastock(request, stock_n):
    gpu
    model_n = 'tdnn'
    model = Predictor(model_n, 32)
    stock = "%s"%stock_n
    todate, x, y, max, min = model.load_transform(stock)
    
    pred = model.predict(x)

    pred = pred.reshape(1,-1)[0]
    for i in range(0, len(pred)):
        pred[i] = (pred[i]*(max-min))+min

    y = y.reshape(1,-1)[0]
    for i in range(0, len(y)):
        y[i] = (y[i]*(max-min))+min 
    
    y = y[~np.isnan(y)]

    pred_tset = pred[0:-7]

    RMSE =round(math.sqrt(MSE(y, pred_tset)), 4)
    EVS = round(explained_variance_score(y, pred_tset), 4)

    yy = y[-28:]

    yyy = pred[-35:]

    a = date.today()

    dateList = []
    for i in range (0, 7):
        d = a + timedelta(days = i)
        d = d.strftime("%d%b")
        dateList.append(d)

    todate = todate.apply(lambda x: x.strftime('%d%b'))
    todate = todate.values.tolist()
    todate = todate[-28:]
    dateList_2 = todate[-28:]

    for dlist in dateList:
        dateList_2.append(dlist)

    labely = "Price"

    chart = get_plot(labely, todate, yy, dateList_2, yyy)
    return render(request, 'djia.html', {"chart": chart, "stock": stock, "RMSE": RMSE, "EVS": EVS})

def djiastockpct(request, stock_n):
    gpu
    model_n = 'tdnn'
    model = Predictor(model_n, 32)
    stock = "%s"%stock_n
    todate, x, y, max, min = model.load_transform(stock)
    
    pred = model.predict(x)

    pred = pred.reshape(1,-1)[0]
    for i in range(0, len(pred)):
        pred[i] = (pred[i]*(max-min))+min

    y = y.reshape(1,-1)[0]
    for i in range(0, len(y)):
        y[i] = (y[i]*(max-min))+min

    y = y[~np.isnan(y)]

    pred_tset = pred[0:-7]

    RMSE =round(math.sqrt(MSE(y, pred_tset)), 4)
    EVS = round(explained_variance_score(y, pred_tset), 4) 

    yy = y[-28:]
    yyy = pred[-35:]
   
    yy = np.diff(yy) / yy[1:] * 100
    yyy = np.diff(yyy) / yyy[1:] * 100

    a = date.today()

    dateList = []
    for i in range (0, 7):
        d = a + timedelta(days = i)
        d = d.strftime("%d%b")
        dateList.append(d)

    todate = todate.apply(lambda x: x.strftime('%d%b'))
    todate = todate.values.tolist()
    todate = todate[-27:]
    dateList_2 = todate[-27:]

    for dlist in dateList:
        dateList_2.append(dlist)

    labely = "%Change"

    chart = get_plot(labely, todate, yy, dateList_2, yyy)
    return render(request, 'djia.html', {"chart": chart, "stock": stock, "RMSE": RMSE, "EVS": EVS})

def setstock(request, stock_n):
    gpu
    model_n = 'tdnn'
    model = Predictor(model_n, 32)
    stock = "%s"%stock_n
    todate, x, y, max, min = model.load_transform(stock)
    
    pred = model.predict(x)

    pred = pred.reshape(1,-1)[0]
    for i in range(0, len(pred)):
        pred[i] = (pred[i]*(max-min))+min

    y = y.reshape(1,-1)[0]
    for i in range(0, len(y)):
        y[i] = (y[i]*(max-min))+min 
    
    y = y[~np.isnan(y)]

    pred_tset = pred[0:-7]

    RMSE =round(math.sqrt(MSE(y, pred_tset)), 4)
    EVS = round(explained_variance_score(y, pred_tset), 4) 

    yy = y[-28:]

    yyy = pred[-35:]

    a = date.today()

    dateList = []
    for i in range (0, 7):
        d = a + timedelta(days = i)
        d = d.strftime("%d%b")
        dateList.append(d)

    todate = todate.apply(lambda x: x.strftime('%d%b'))
    todate = todate.values.tolist()
    todate = todate[-28:]
    dateList_2 = todate[-28:]

    for dlist in dateList:
        dateList_2.append(dlist)
    
    labely = "Price"

    chart = get_plot(labely, todate, yy, dateList_2, yyy)
    return render(request, 'set100.html', {"chart": chart, "stock": stock, "RMSE": RMSE, "EVS": EVS})

def setstockpct(request, stock_n):
    gpu
    model_n = 'tdnn'
    model = Predictor(model_n, 32)
    stock = "%s"%stock_n
    todate, x, y, max, min = model.load_transform(stock)
    
    pred = model.predict(x)

    pred = pred.reshape(1,-1)[0]
    for i in range(0, len(pred)):
        pred[i] = (pred[i]*(max-min))+min

    y = y.reshape(1,-1)[0]
    for i in range(0, len(y)):
        y[i] = (y[i]*(max-min))+min 

    y = y[~np.isnan(y)]

    pred_tset = pred[0:-7]

    RMSE =round(math.sqrt(MSE(y, pred_tset)), 4)
    EVS = round(explained_variance_score(y, pred_tset), 4) 

    yy = y[-28:]
    yyy = pred[-35:]
   
    yy = np.diff(yy) / yy[1:] * 100
    yyy = np.diff(yyy) / yyy[1:] * 100

    a = date.today()

    dateList = []
    for i in range (0, 7):
        d = a + timedelta(days = i)
        d = d.strftime("%d%b")
        dateList.append(d)

    todate = todate.apply(lambda x: x.strftime('%d%b'))
    todate = todate.values.tolist()
    todate = todate[-27:]
    dateList_2 = todate[-27:]

    for dlist in dateList:
        dateList_2.append(dlist)

    labely = "Percent Change"

    chart = get_plot(labely, todate, yy, dateList_2, yyy)
    return render(request, 'set100.html', {"chart": chart, "stock": stock, "RMSE": RMSE, "EVS": EVS})