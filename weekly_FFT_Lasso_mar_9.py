# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.fftpack as fft
from matplotlib import rc
from math import pi as pi
import sklearn.linear_model as sklm
from sklearn.cross_validation import KFold


def read_data():
    """
    データを読み込む
    :return:data, unit
    """
    data = pd.read_csv("~/PycharmProjects/MATSUO/daily_category_sales_quantity.csv", parse_dates=0, header=None)
    data.columns = ["date", "catg_cd", "catg", "sales", "qty"]
    return data

def make_data(code):
    """
    codeごとに学習データを作成plt.plot(fftYlist)
    plt.show()
    qty_dataに格納
    """
    df_by_code = data[data["catg_cd"] == code]
    df_by_code = df_by_code.set_index("date")
    df_by_code.index = pd.to_datetime(df_by_code.index)

    cteg_data = df_by_code[(df_by_code.index.year >= 2008) & (df_by_code.index.year <= 2013)]
    weekly_catg_data = cteg_data["qty"].resample('W', how='sum', closed='left', label='left')
    qty_data = weekly_catg_data.drop(list(weekly_catg_data.index)[0])
    qty_data = qty_data.reset_index()
    qty_train = qty_data[:261]

    qty_test = qty_data["qty"][-52:-1]
    qty_test = qty_test.reset_index()

    return qty_data, qty_train, qty_test


def forecast1():
    """
    ４週ずつ予測を繰り返す。
    この関数では、ログを取り、トレンド項を削除する。
    :param i:
    :return:
    """
    qty_train["logged_qty"] = qty_train["qty"].apply(np.log)
    x = list(qty_train.index)
    x = sm.add_constant(x)
    y = qty_train["logged_qty"]
    model = sm.OLS(y, x)
    results = model.fit()
    intercept, x1 = results.params
    pred = qty_train.index * x1 + intercept
    Y = qty_train["logged_qty"] - pred
    lenY = len(Y)
    Lasso_X, fftY, freqs, power, phase, lenY = FFT(Y,lenY)
    fftYlist = list(fftY)
    print("Mean", np.mean(np.abs(fftY)))
    print("variance",np.var(fftY))
    #wfftY = fft.fft(np.abs(fftY[:156]))
    #wfftYlist = list(wfftY)
    #plt.plot(wfftYlist)

    qty_forecast = do_lasso(Y, Lasso_X, fftY, freqs, power, phase, lenY)
    qty_forecast = qty_forecast + intercept + [i * x1 for i in range(0, lenY+51)]
    qty_forecast = np.exp(qty_forecast)

    qty_forecast = qty_forecast[-51:]
    return qty_forecast

def FFT(Y,lenY):
    """
    フーリエ変換
    :param Y:
    :param lenY:
    :return:
    """
    fftY = fft.fft(Y)
    freqs = fft.fftfreq(len(Y))
    power = np.abs(fftY)
    phase = [np.arctan2(float(c.imag), float(c.real)) for c in fftY]

    Lasso_X = make_lassoX(fftY, freqs, power, phase, lenY)

    return Lasso_X, fftY, freqs, power, phase, lenY

def make_lassoX(fftY, freqs, power, phase, lenY):
    """
    周波数ごとに波を作成しLasso_Xを作成する
    :param fftY:
    :param freqs:
    :param power:
    :param phase:
    :param lenY:
    :return: Lasso_X
    """
    for po, fr, ph in zip(power, freqs, phase):
        average = []
        i = 0
        for t in range(0, lenY):
            average.append(po * np.cos((2 * pi * fr) * t + ph) / (lenY))
            i += 1
        Lasso_X[fr] = average

    return Lasso_X

def do_lasso(Y, Lasso_X, fftY, freqs, power, phase, lenY):

    #Lasso_X.plot()
    #plt.show()

    Lasso_model = sklm.Lasso(alpha=0.0001)
    results = Lasso_model.fit(Lasso_X, Y)

    print(results.coef_)
    print(results.score(Lasso_X,Y))
    #plt.plot((results.coef_ * power))
    #plt.show()
    qty_forecast = []
    for t in range(0, lenY+51):
        average = 0
        for coef, po, fr, ph in zip(results.coef_, power, freqs, phase):
            average += coef * po * np.cos((2 * pi * fr) * t + ph) / (lenY+51)
        qty_forecast.append(average)

    return qty_forecast

def evaluate():
    #ウィークリー評価
    wmape = sum(abs(forecast-qty_test["qty"])/qty_test["qty"]) / 51 * 100

    reg = forecast - qty_test["qty"]
    #print(reg)
    preg = list(reg)
    mreg = list(reg)
    i=0
    for i in range(0,51):
        if reg[i]>=0:
            mreg[i] = 0
        elif reg[i]<0:
            preg[i] = 0
        i +=1
    preg = np.array(preg)
    mreg = np.array(mreg)
    wpmape = sum(abs(preg[0:51])/qty_test["qty"][0:51]) / 51 * 100
    wmmape = sum(abs(mreg[0:51])/qty_test["qty"][0:51]) / 51 * 100


    test = qty_test["qty"][0:51]
    print("weekly")
    print("MAPE", wmape)
    print("PlusMAPE", wpmape)
    print("MinusMAPE", wmmape)

    evaluation = [wmape, wpmape, wmmape]



    return evaluation,test

if __name__ == '__main__':
    data = read_data()
    codeList = [1101,1102,1103, 1104, 1105, 1107, 1108, 1109, 1110, 1111, 1220, 1221, 1222, 1223, 1224, 1226, 1227, 1228, 5330, 5331, 5332]
    submit = pd.DataFrame(None)
    evaluation_submit = pd.DataFrame(None)
    tests = pd.DataFrame(None)

    k = 1
    for item in codeList:
        qty_data, qty_train, qty_test = make_data(code=item)
        #qty_train.plot()
        #plt.xlabel("Week")
        #plt.ylabel("Quantity")
        #plt.show()
        forecast = []
        print(k)
        Lasso_X = pd.DataFrame(None)
        forecast = forecast1()
        submit[k] = forecast
        evaluation,test = evaluate()
        evaluation_submit[k] = evaluation
        tests[k] = test
        k += 1
    submit.columns = codeList
    evaluation_submit.columns = codeList
    tests.columns = codeList
    evaluationList = ["wmape","wpmape","wmmape"]
    evaluation_submit.index = evaluationList

    submit.to_csv('forecast_3_29.csv')
    evaluation_submit.to_csv('evaluation_3_19.csv')
    tests.to_csv('test_3_19.csv')
