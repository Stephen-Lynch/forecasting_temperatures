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

tr_start, tr_end = '2016', '2017'
te_start = '2017-01-01'
tra = den_2015[tr_start:tr_end]
tes = den_2015[te_start:]

model = pickle.load(open('forecasting.sav', 'rb'))

if __name__ == '__main__':
    forecast = model.predict(start = len(tra),
                              end = len(tra) + 104,
                              typ = 'levels').rename('Forecast')

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(den_2015['2014':'2018'])
    ax.plot(forecast)