import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('ggplot')
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima
import warnings 
warnings.filterwarnings("ignore") 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
from eda import df_create, datetime_month, drop_month_year


def create_cities(cities):
    '''
    Creates a list of pandas dataframes based on cities

    Args:
        cities - List of Strings

    Returns:
        List of pandas dataframes that are cities
    '''
    citylst = []
    for city in cities:
        city = city_temp.loc[city_temp['City'] == city]
        city = city.drop(['Region', 'Country', 'City', 'Month', 'Day', 'Year'], axis = 1).groupby('Date').mean()
        city = city['AvgTemperature']
        city = city.resample('W-MON').mean()
        citylst.append(city)
    return citylst

def create_time_series(cities):
    '''
    Creates a liss of pandas series based on the cities

    Args:
        cities - List of Cities Pandas df
    
    Returns:
        List of time series
    '''
    citylst = []
    for city in cities:
        city = df_create(city_temp, ['Region', 'Country', 'City', 'Day', 'Month', 'Year'], ['Date'], reset_index = False)
        city = city['AvgTemperature']
        city = city.resample('W-MON')
        citylst.append(city)
    return citylst

def create_model(city, p, d, q, P, D, Q, m):
    '''
    Creates a Sarimax and returns your model and forecast on full dataset

    Args:
        city - series of a city
    
    Returns:
        Model fit to train data and forecast with city data
    '''
    train1 = city.iloc[:len(city)- 52]
    test1 = city.iloc[len(city) - 52:]

    model = SARIMAX(train1,
                    order = (p, d ,q),
                    seasonal_order =(P, D, Q, m) )
    result = model.fit()

    model = SARIMAX(train1,  
                        order = (p, d, q),  
                        seasonal_order =(P, D, Q, m)) 
    result1 = model.fit()
    

    forecast = result1.predict(start = len(city) -  104 ,
                              end = (len(city)) + 8,
                              typ = 'levels').rename('Forecast')
    return result, forecast

def create_predictions_plot(axs, cities, predictions):
    '''
    Creates a plot based on cities and predictions

    ARGS:
        axs - list of ax plots
        cities - list of time series
        predictions - list of model_predictions
    
    Return:
        Plot if plt.show() is down below
    '''
    axs[0].plot(predictions[0], label= 'prediction')
    axs[0].plot(cities[0].iloc[len(cities[0]) - 52:], label='Actual')
    axs[0].set_title('Nigeria, Niamey Average Temperature Weekly')
    axs[1].plot(predictions[1], label= 'prediction')
    axs[1].plot(cities[1].iloc[len(cities[1]) - 52:], label='Actual')
    axs[1].set_title('Kuwait, Kuwait Average Temperature Weekly')
    axs[1].legend()
    axs[2].plot(predictions[2], label= 'prediction')
    axs[2].plot(cities[2].iloc[len(cities[2]) - 52:], label='Actual')
    axs[2].set_title('UAE, Dubai Average Temperature Weekly')


## Instantiating the needed data to work with my SARIMAX models
city_temp = pd.read_csv('data/cleaned_city_temps.csv')
city_temp = city_temp.drop('Unnamed: 0', axis = 1)
city_temp['Date'] = pd.to_datetime(city_temp[['Year', 'Month', 'Day']])


# Creating the 3 cities I'm going to model

cities = ['Niamey', 'Kuwait', 'Dubai']
cities = create_cities(cities)
# cities = create_time_series(cities)

niamey_mod, niamey_forecast = create_model(cities[0], 1, 0, 0, 2, 1, 0, 12)
kuwait_mod, kuwait_forecast = create_model(cities[1], 1 ,0, 2, 2, 1, 0, 12)
dubai_mod, dubai_forecast = create_model(cities[2], 1, 0, 0, 2, 1, 0, 12)


#Using AutoArima but it looks like a diff of 2 periods with a log of 1 will be best
# stepwise_fit = auto_arima(cities[2].dropna(), start_p = 1, start_q = 1, max_p = 2, max_q = 2, m = 4, start_P = 0, seasonal = True, d = None, D=1, trace = True, error_action = 'ignore', suppress_warning = True, stepwise = True)


if __name__ == '__main__':
    # print(cities)

    #Plots prediction values
    fig, axs = plt.subplots(3, figsize=(20, 4), dpi = 200)
    # start = len(cities[0].iloc[:len(cities[0])- 52])
    # end = len(cities[0].iloc[:len(cities[0])- 52]) + len(cities[0].iloc[len(cities[0]) - 52:]) - 1
    # predictions1 = niamey_mod.predict(start, end, typ='levels').rename('Predictions')
    # predictions2 = kuwait_mod.predict(start, end, typ='levels').rename('Predictions')
    # predictions3 = dubai_mod.predict(start, end, typ='levels').rename('Predictions')
    # predlst = [predictions1, predictions2, predictions3]
    # create_predictions_plot(axs, cities, predlst)
    ## Creates a plot based on forecasting data
    axs[0].plot(cities[0].iloc[-104:], label = 'Avg Temp', color = 'b')
    axs[0].plot(niamey_forecast, label ='Forecast', color ='y', linewidth=2)
    axs[0].set_title('Nigeria, Niamey Average Temperature Weekly')
    axs[0].legend(loc='upper left')
    axs[1].plot(cities[1].iloc[-104:], label = 'Avg Temp', color = 'b')
    axs[1].plot(kuwait_forecast, label ='Forecast', color ='y', linewidth=2)
    axs[1].set_title('Kuwait, Kuwait Average Temperature Weekly')
    axs[1].legend(loc='upper left')
    axs[2].plot(cities[2].iloc[-104:], label = 'Avg Temp', color = 'b')
    axs[2].plot(dubai_forecast, label ='Forecast', color ='y', linewidth=2)
    axs[2].legend(loc='upper left')
    axs[2].set_title('UAE, Dubai Average Temperature Weekly')
    fig.tight_layout()
    fig.tight_layout()
    plt.xticks(fontsize = 16)
    plt.show()