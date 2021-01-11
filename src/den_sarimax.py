import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('ggplot')
import statsmodels.api as sm
import warnings 
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima
import warnings 
warnings.filterwarnings("ignore") 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_error as MAE
import pickle


def forecast_example(forecast):
    lst = []
    t = 0
    swim = 0
    shorts = 0
    sleeves = 0
    pants = 0
    H_jack = 0
    L_jack = 0
    H_pants = 0
    for temp in forecast:
        if temp > 100:
            t += 1
            swim += 1
            shorts += 1
        elif temp <= 99.9999 and temp >= 60:
            t += 1
            swim += 1
            shorts += 1
        elif temp <=59.999 and temp >= 30:
            sleeves += 1
            pants += 1
        elif temp <=29.999 and temp >= 10:
            L_jack += 1
            sleeves += 1
            pants += 1
        elif temp <= 10:
            H_jack +=1
            sleeves += 1
            H_pants += 1
    lst.extend((t, swim, shorts, sleeves, pants, H_jack, L_jack))
    return lst

city_temps = pd.read_csv('data/col_city_temps.csv')
city_temps = city_temps.drop('Unnamed: 0', axis = 1)

col_temps =  city_temps.drop(['Country', 'State', 'City', 'Region'], axis = 1)
col_temps['Date'] = pd.to_datetime(col_temps[['Year', 'Month', 'Day']])
col_temps = col_temps.drop(['Month', 'Day', 'Year'], axis = 1)
col_temps = col_temps.groupby('Date').mean()
col_temps = col_temps.resample('W-MON').mean()

den = city_temps[city_temps['City'] == 'Denver']
den_2015 = den[(den['Year'] > 2005)]
den_2015 =  den_2015.drop(['Country', 'State', 'City', 'Region'], axis = 1)
den_2015['Date'] = pd.to_datetime(den_2015[['Year', 'Month', 'Day']])
den = den_2015
den_2015 = den_2015.drop(['Month', 'Day', 'Year'], axis = 1)
den_2015 = den_2015.groupby('Date').mean()

tr_start, tr_end = '2019-01-01', '2020-01-01'
te_start = '2018-01-01'
tra = den_2015[tr_start:tr_end]
tes = den_2015[te_start:]

model = SARIMAX(tra, order = (1, 1, 2), seasonal_order = (2, 1, 0, 26)).fit()
# model = SARIMAX(tra, order = (2, 0, 2), seasonal_order = (2, 1, 2, 90)).fit()

forecast = model.predict(start = len(tra) ,
                              end = len(tra) + 13,
                              typ = 'levels').rename('Forecast')




# stepwise_fit = auto_arima(tra, start_p = 1, start_q = 1, max_p = 2, max_q = 2, m = 26, start_P = 0, seasonal = True, d = None, D=1)
# , trace = True, error_action = 'ignore', suppress_warning = True, stepwise = True)
if __name__ == '__main__':
    print(MAE(den_2015['2020-01-02':"2020-01-15"].values, forecast))
    fig, ax = plt.subplots(figsize=(20, 12), dpi = 200)
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    ax.plot(den_2015['2019-12-23':"2020-01-15"], label= 'Actual Temperature', c = "b")
    ax.plot(forecast, label = 'Forecasted Temperature', c = "black")
    ax.set_xlabel("Date", fontsize = 20, c = 'black')
    ax.set_ylabel("°F", fontsize = 20, c= 'black')
    ax.set_title('Denver Forecasted Temperatures 14 Days', fontsize= 24)
    ax.tick_params(axis='x', colors = 'black')
    ax.tick_params(axis='y', colors = 'black')
    plt.xticks(rotation=20)
    fig.tight_layout()
    ax.legend(fontsize = 18, loc='upper left')
    plt.show()

    # fig, axs = plt.subplots(2, figsize=(20, 8), dpi = 200)
    # fig = sm.graphics.tsa.plot_acf(tra.diff().dropna(), lags =  180, ax = axs[0])
    # fig = sm.graphics.tsa.plot_pacf(tra.diff().dropna(), lags = 180, ax = axs[1], method='ywmle')
    # plt.show()

    # lst = forecast_example(forecast)

    # t, swim, shorts, sleeves, pants, H_jack, L_jack, H_pants

    # string = ""
    # for idx, item in enumerate(lst):
    #     if item == 0:
    #         continue
    #     elif idx == 0:
    #         string += "Short Sleeve: {}, ".format(item)
    #     elif idx == 1:
    #         string += "Swim Wear: {}, ".format(item)
    #     elif idx == 2:
    #         string += "Shorts: {}, ".format(item)
    #     elif idx == 3:
    #         string += "Sleeves: {}, ".format(item)
    #     elif idx == 4:
    #         string += "Pants: {}, ".format(item)
    #     elif idx == 5:
    #         string +="Heavy Jacket: {}, ".format(item)
    #     elif idx == 6:
    #         string +="Light Jacket: {}".format(item)

