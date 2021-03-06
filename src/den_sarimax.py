import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('ggplot')
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima
import warnings 
warnings.filterwarnings("ignore") 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error as MAE
import pickle


def forecast_example(forecast):
    '''
    Keeps track of how many times forecasted temperatures landed within a specific range

    Args:
        forecast - Array object

    Returns:
        A list that keeps track of temperature ranges
    '''
    lst = []
    high = 0
    medHigh = 0
    med = 0
    medLow = 0
    Low = 0
    for temp in forecast:
        if temp > 100:
            high += 1
        elif temp <= 99.9999 and temp >= 60:
            medHigh += 1
        elif temp <=59.999 and temp >= 30:
            med += 1
        elif temp <=29.999 and temp >= 10:
            medLow += 1
        elif temp <= 10:
            low += 1
    lst.extend((high, medHigh, med, medLow, Low))
    return lst

def time_series(city, df):
    '''
    Creates a time series based on a city name

    Args:
        city - String
        df - Dataframe object

    Returns:
        Dataframe Time Series
    '''
    cty = df[df['City'] == city]
    cty = cty[cty['Year'] > 2005]
    cty = cty.drop(['Country', 'State', 'City', 'Region'], axis = 1)
    cty['Date'] = pd.to_datetime(cty[['Year', 'Month', 'Day']])
    cty = cty.drop(['Month', 'Day', 'Year'], axis = 1)
    cty = cty.groupby('Date').mean()
    return cty




if __name__ == '__main__':
    city_temps = pd.read_csv('data/col_city_temps.csv')
    city_temps = city_temps.drop('Unnamed: 0', axis = 1)

    ##Creating and fitting the model
    den_2015 = time_series('Denver', city_temps)

    tr_start, tr_end = '2019-01-01', '2020-01-01'
    te_start = '2018-01-01'
    tra = den_2015[tr_start:tr_end]
    tes = den_2015[te_start:]

    model = SARIMAX(tra, order = (1, 1, 2), seasonal_order = (2, 1, 0, 26)).fit()

    forecast = model.predict(start = len(tra) ,
                              end = len(tra) + 13,
                              typ = 'levels').rename('Forecast')

    # Creating Time Series and Forecasting Plot
    print(MAE(den_2015['2020-01-02':"2020-01-15"].values, forecast))
    fig, ax = plt.subplots(figsize=(20, 12), dpi = 200)
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    ax.plot(den_2015['2019-12-23':"2020-01-15"], label= 'Actual Temperature', c = "b", 
            linewidth = 4.0)
    ax.plot(forecast, label = 'Forecasted Temperature', c = "black",
            linewidth = 4.0, marker = 'o')
    ax.set_xlabel("Date", fontsize = 20, c = 'black')
    ax.set_ylabel("°F", fontsize = 20, c= 'black')
    ax.set_title('Denver Forecasted Temperatures 14 Days', fontsize= 24)
    ax.tick_params(axis='x', colors = 'black')
    ax.tick_params(axis='y', colors = 'black')
    plt.xticks(rotation=20)
    fig.tight_layout()
    ax.legend(fontsize = 18, loc='upper left')
    # ax.axhspan(50, 30, color = 'lime', alpha = .2)
    # ax.axhspan(30,10, color = 'blue', alpha = .2)
    ax.set_ylim([10,50])
    plt.show()

    ## Creating acf plots
    # fig, axs = plt.subplots(2, figsize=(20, 8), dpi = 200)
    # fig = sm.graphics.tsa.plot_acf(tra.diff().dropna(), lags =  180, ax = axs[0])
    # fig = sm.graphics.tsa.plot_pacf(tra.diff().dropna(), lags = 180, ax = axs[1], method='ywmle')
    # plt.show()

    
    
    ## Creating printout example of what forecasting with my program would look like
    # lst = forecast_example(forecast)
    # string = "\n\n\nBased On the Temperatures Between the Dates 1/2/2020 and 1/15/2020:\n"
    # suggestion = ""
    # for idx, item in enumerate(lst):
    #     if item == 0:
    #         continue
    #     elif idx == 0:
    #         string += "There are going to be {} days that are above 100 Degrees.\n".format(item)
    #     elif idx == 1:
    #         string += "Therea re going to be {} days that are between 99 and 60 degrees".format(item)
    #     elif idx == 2:
    #         string += "There are going to be {} days that are between 59 and 30 Degrees.\n".format(item)
    #         if lst[idx] == max(lst):
    #             suggestion = "Make sure to pack a Heavy Jacket and some long sleeved shirts!"
    #     elif idx == 3:
    #         string += "There are going to be {} days that are between 29 and 10 Degrees.\n".format(item)
    #         if lst[idx] == max(lst):
    #             suggestion = "Make sure to bring a Medium Jacket and some Hoodies."
    #     elif idx == 4:
    #         string += "There are going to be {} days that are below 10 Degrees.\n".format(item)
    #         if lst[idx] == max(lst):
    #             suggestion = "Make sure to bring a Heavy Jacket and either some Long Sleeved Shirts or some Sweaters."

    # print(string + suggestion + "\n\n\n")