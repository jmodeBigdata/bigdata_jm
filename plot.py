# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt

def read_forecasts():
    FFT = pd.read_csv("~/PycharmProjects/MATSUO/forecast_3_29.csv",index_col=0, header=0, names=codeList)
    ARIMA = pd.read_csv("~/PycharmProjects/MATSUO/ARIMA.csv", index_col=0, header=0, names=codeList)
    ACTUAL = pd.read_csv("~/PycharmProjects/MATSUO/test_3_19.csv", index_col=0, header=0, names=codeList)

    return FFT, ARIMA, ACTUAL




if __name__ == '__main__':
    codeList = [1101,1102,1103, 1104, 1105, 1107, 1108, 1109, 1110, 1111, 1220, 1221, 1222, 1223, 1224, 1226, 1227, 1228, 5330, 5331, 5332]
    FFTf, ARIMAf, ACTUALf = read_forecasts()

    MAPE = pd.DataFrame(None)

    i = 1

    for item in codeList:
        FFT = FFTf[item]
        SARIMA = ARIMAf[item]
        ACTUAL = ACTUALf[item]

        FourierTransform_MAPE = sum(abs(FFT - ACTUAL)/ACTUAL) / 51 * 100
        SARIMA_MAPE = sum(abs(SARIMA - ACTUAL)/ACTUAL) / 51 * 100
        FourierTransform_RMSE = (sum((FFT-ACTUAL)**2)/51)**(1/2)
        SARIMA_RMSE = ((sum((SARIMA -ACTUAL)**2))/51)**(1/2)
        MAPE[i] = [item, FourierTransform_MAPE,SARIMA_MAPE,FourierTransform_RMSE,SARIMA_RMSE]

        plt.figure()
        FFT.plot(label="Forecast")
        SARIMA.plot(label="SARIMA")
        ACTUAL.plot(label="Actual")

        plt.legend()
        plt.xlabel("Week")
        plt.ylabel("Quantity")
        plt.show()

        i += 1

    MAPE.to_csv("MAPE_RMSE_3_19.csv",header=False)
