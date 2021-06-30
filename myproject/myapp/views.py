from myapp.preprocessstock import preprocess
from django.shortcuts import render
from django.http import HttpResponse
from .predictor import Predictor
from .preprocessstock import preprocess
from .plot import get_plot
from .pathModel import pathModel
from .runModel import runModel
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import json
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
    stock = "%s"%stock_n
    model_n = 'tdnn'
    actual, trend_tdnn, predict_tdnn, dateList, todate, todate_p_tdnn, df_tndd = runModel(model_n, stock)

    model_n = 'tdnn_pso'
    actual, trend_tdnn_pso, predict_tdnn_pso, dateList, todate, todate_p_tdnn_pso, df_tdnn_pso = runModel(model_n, stock)

    model_n = 'rf'
    actual, trend_rf, predict_rf, dateList, todate, todate_p_rf, df_rf = runModel(model_n, stock)
    
    model_n = 'svm'
    actual, trend_svm, predict_svm, dateList, todate, todate_p_svm, df_svm = runModel(model_n, stock)
    
    df = pd.DataFrame(columns=list(['Model', 'RMSE', 'EVS']))
    df = df.append(df_tndd)
    df = df.append(df_tdnn_pso)
    df = df.append(df_rf)
    df = df.append(df_svm)
    json_records = df.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)

    labely = "Price"

    chart = get_plot(labely, todate, actual, todate_p_tdnn, trend_tdnn, dateList, predict_tdnn, todate_p_tdnn_pso, trend_tdnn_pso, 
    dateList, predict_tdnn_pso, todate_p_rf, trend_rf, dateList, predict_rf, todate_p_svm, trend_svm, dateList, predict_svm)
    return render(request, 'djia.html', {"chart": chart, "stock": stock, 'd': data})

def djiastockpct(request, stock_n):
    gpu
    stock = "%s"%stock_n
    model_n = 'tdnn'
    actual, trend_tdnn, predict_tdnn, dateList, todate, todate_p_tdnn, df_tndd = runModel(model_n, stock, type_y=False)

    model_n = 'tdnn_pso'
    actual, trend_tdnn_pso, predict_tdnn_pso, dateList, todate, todate_p_tdnn_pso, df_tdnn_pso = runModel(model_n, stock, type_y=False)

    model_n = 'rf'
    actual, trend_rf, predict_rf, dateList, todate, todate_p_rf, df_rf = runModel(model_n, stock, type_y=False)
    
    model_n = 'svm'
    actual, trend_svm, predict_svm, dateList, todate, todate_p_svm, df_svm = runModel(model_n, stock, type_y=False)
    
    df = pd.DataFrame(columns=list(['Model', 'RMSE', 'EVS']))
    df = df.append(df_tndd)
    df = df.append(df_tdnn_pso)
    df = df.append(df_rf)
    df = df.append(df_svm)
    json_records = df.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)

    labely = "%Change"

    chart = get_plot(labely, todate, actual, todate_p_tdnn, trend_tdnn, dateList, predict_tdnn, todate_p_tdnn_pso, trend_tdnn_pso, 
    dateList, predict_tdnn_pso, todate_p_rf, trend_rf, dateList, predict_rf, todate_p_svm, trend_svm, dateList, predict_svm)
    return render(request, 'djia.html', {"chart": chart, "stock": stock, 'd': data})

def setstock(request, stock_n):
    gpu
    stock = "%s"%stock_n
    model_n = 'tdnn'
    actual, trend_tdnn, predict_tdnn, dateList, todate, todate_p_tdnn, df_tndd = runModel(model_n, stock)

    model_n = 'tdnn_pso'
    actual, trend_tdnn_pso, predict_tdnn_pso, dateList, todate, todate_p_tdnn_pso, df_tdnn_pso = runModel(model_n, stock)

    model_n = 'rf'
    actual, trend_rf, predict_rf, dateList, todate, todate_p_rf, df_rf = runModel(model_n, stock)
    
    model_n = 'svm'
    actual, trend_svm, predict_svm, dateList, todate, todate_p_svm, df_svm = runModel(model_n, stock)
    
    df = pd.DataFrame(columns=list(['Model', 'RMSE', 'EVS']))
    df = df.append(df_tndd)
    df = df.append(df_tdnn_pso)
    df = df.append(df_rf)
    df = df.append(df_svm)
    json_records = df.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)

    labely = "Price"

    chart = get_plot(labely, todate, actual, todate_p_tdnn, trend_tdnn, dateList, predict_tdnn, todate_p_tdnn_pso, trend_tdnn_pso, 
    dateList, predict_tdnn_pso, todate_p_rf, trend_rf, dateList, predict_rf, todate_p_svm, trend_svm, dateList, predict_svm)
    return render(request, 'set100.html', {"chart": chart, "stock": stock, 'd': data})

def setstockpct(request, stock_n):
    gpu
    stock = "%s"%stock_n
    model_n = 'tdnn'
    actual, trend_tdnn, predict_tdnn, dateList, todate, todate_p_tdnn, df_tndd = runModel(model_n, stock, type_y=False)

    model_n = 'tdnn_pso'
    actual, trend_tdnn_pso, predict_tdnn_pso, dateList, todate, todate_p_tdnn_pso, df_tdnn_pso = runModel(model_n, stock, type_y=False)

    model_n = 'rf'
    actual, trend_rf, predict_rf, dateList, todate, todate_p_rf, df_rf = runModel(model_n, stock, type_y=False)
    
    model_n = 'svm'
    actual, trend_svm, predict_svm, dateList, todate, todate_p_svm, df_svm = runModel(model_n, stock, type_y=False)
    
    df = pd.DataFrame(columns=list(['Model', 'RMSE', 'EVS']))
    df = df.append(df_tndd)
    df = df.append(df_tdnn_pso)
    df = df.append(df_rf)
    df = df.append(df_svm)
    json_records = df.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)

    labely = "%Change"

    chart = get_plot(labely, todate, actual, todate_p_tdnn, trend_tdnn, dateList, predict_tdnn, todate_p_tdnn_pso, trend_tdnn_pso, 
    dateList, predict_tdnn_pso, todate_p_rf, trend_rf, dateList, predict_rf, todate_p_svm, trend_svm, dateList, predict_svm)
    return render(request, 'set100.html', {"chart": chart, "stock": stock, 'd': data})