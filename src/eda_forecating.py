import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('ggplot')
import statsmodels.api as sm
import warnings 
warnings.filterwarnings("ignore") 
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima
import warnings 
warnings.filterwarnings("ignore") 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
import pickle

def plot_seasonal_decomposition(axs, series, sd):
    '''
    Plots a time series decomposition based on a series and a seasonal
    Decompose

    Args:
        axs - A graph on which to plot the decomposition
        series - a Pandas series like object
        sd - a Seasonal Decomposition of said object

    Returns:
        A plot of if plt.show() is in if __name__ == __main__
    '''
    axs[0].plot(series, c='b')
    axs[0].set_title("Raw Series", fontsize = 24)
    axs[0].set_ylabel('Degrees', fontsize = 20)
    axs[1].plot(sd.trend, c='b')
    axs[1].set_title("Trend Component $T_t$", fontsize = 24)
    axs[1].set_ylabel('Degrees', fontsize = 20)
    axs[2].plot(sd.seasonal, c='b')
    axs[2].set_title("Seasonal Component $S_t$", fontsize = 24)
    axs[2].set_ylabel('Degrees', fontsize = 20)
    axs[2].set_xlabel('Date', fontsize = 20)

city_temps = pd.read_csv('data/col_city_temps.csv')
city_temps = city_temps.drop("Unnamed: 0", axis = 1)
city_temp = pd.read_csv('data/city_temperature.csv')

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

san = city_temp[city_temp['City'] == 'Austin']
san = san[(san['Year'] > 2005) & (san['State'] == 'Texas')]
san =  san.drop(['Country', 'State', 'City', 'Region'], axis = 1)
san['Date'] = pd.to_datetime(san[['Year', 'Month', 'Day']])
san = san.drop(['Month', 'Day', 'Year'], axis = 1)
san = san.groupby('Date').mean()

# tr_start, tr_end = '2016', '2017'
# te_start = '2017-01-01'
# tra = den_2015[tr_start:tr_end]
# tes = den_2015[te_start:]

result = seasonal_decompose(den_2015['2010': ], model='additive', freq=365)

if __name__ == '__main__':
    fig, axs = plt.subplots(2, figsize=(24, 8), dpi = 200)
    
    axs[0].plot(den_2015['2015'], c='b')
    axs[0].set_title('Daily Denver Temperature', fontsize = 24)
    axs[0].set_ylim([0, 100])
    axs[1].plot(san['2015'], c='r')
    axs[1].set_title('Daily Austin Temperature', fontsize = 24)
    axs[1].set_ylabel("°F", fontsize = 20)
    axs[1].set_xlabel("Date", fontsize = 24)
    axs[1].set_ylim([0, 100])
    axs[0].set_ylabel("°F", fontsize = 20)
    plt.setp(axs[0].get_xticklabels(), fontsize=20)
    plt.setp(axs[0].get_yticklabels(), fontsize=20)
    plt.setp(axs[1].get_xticklabels(), fontsize=20)
    plt.setp(axs[1].get_yticklabels(), fontsize=20)
    plt.show()

    # fig, axs = plt.subplots(3, figsize = (24, 8), dpi = 200)
    # plt.setp(axs[0].get_xticklabels(), fontsize=20)
    # plt.setp(axs[0].get_yticklabels(), fontsize=20)
    # plt.setp(axs[1].get_xticklabels(), fontsize=20)
    # plt.setp(axs[1].get_yticklabels(), fontsize=20)
    # plt.setp(axs[2].get_xticklabels(), fontsize=20)
    # plt.setp(axs[2].get_yticklabels(), fontsize=20)
    # fig.tight_layout()
    # plot_seasonal_decomposition(axs, den_2015['2010': ], result)
    # plt.show()
    