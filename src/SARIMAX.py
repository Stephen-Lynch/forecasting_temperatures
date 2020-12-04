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
        city = city_temp_train.loc[city_temp_train['City'] == city]
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
        city = df_create(city, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
        city['Date'] = datetime_month(city)
        city = drop_month_year(city)
        citylst.append(city)
    return citylst

def create_model(city):
    '''
    Creates a Sarimax and returns your model and forecast on full dataset

    Args:
        city - series of a city
    
    Returns:
        Model fit to train data and forecast with city data
    '''
    train1 = city.iloc[:len(city)- 24]
    test1 = city.iloc[len(city) - 24:]

    model = SARIMAX(train1,
                    order = (1, 0 ,0),
                    seasonal_order =(2, 1, 2, 4) )
    result = model.fit()

    model = SARIMAX(city,  
                        order = (1, 0, 0),  
                        seasonal_order =(2, 1, 2, 4)) 
    result1 = model.fit()

    forecast = result1.predict(start = len(city) - 12  ,
                              end = (len(city)) + 12,
                              typ = 'levels').rename('Forecast')
    return result, forecast


## Instantiating the needed data to work with my SARIMAX models
city_temp = pd.read_csv('data/cleaned_city_temps.csv')
city_temp = city_temp.drop('Unnamed: 0', axis = 1)
city_temp_train = city_temp.loc[city_temp['Year'] < 2017]
city_temp_ho = city_temp.loc[(city_temp['Year'] > 2016) 
                            * (city_temp['Year'] < 2019)]



# Creating the 3 cities I'm going to model

cities = ['Niamey', 'Kuwait', 'Dubai']
cities = create_cities(cities)
cities = create_time_series(cities)

niamey_mod, niamey_forecast = create_model(cities[0])
kuwait_mod, kuwait_forecast = create_model(cities[1])
dubai_mod, dubai_forecast = create_model(cities[2])

# Using AutoArima but it looks like a diff of 2 periods with a log of 1 will be best
# stepwise_fit = auto_arima(dubai['AvgTemperature'], start_p = 1, start_q = 1, max_p = 5, max_q = 5, m = 4,
#                           start_P = 0, seasonal = True, d = None, D=1, trace = True, error_action = 'ignore',
#                           suppress_warning = True, stepwise = True)


if __name__ == '__main__':
    # print(cities)

    ##Plots prediction values
    # fig, axs = plt.subplots(3, figsize=(20, 4), dpi = 200)
    # start = len(cities[0].iloc[:len(cities[0])- 24])
    # end = len(cities[0].iloc[:len(cities[0])- 24]) + len(cities[0].iloc[len(cities[0]) - 24:]) - 1

    # predictions1 = niamey_mod.predict(start, end, typ='levels').rename('Predictions')
    # predictions2 = kuwait_mod.predict(start, end, typ='levels').rename('Predictions')
    # predictions3 = dubai_mod.predict(start, end, typ='levels').rename('Predictions')

    # axs[0].plot(predictions1, label= 'prediction')
    # axs[0].plot(cities[0].iloc[len(cities[0]) - 24:], label='Actual')
    # axs[0].set_title('Nigeria, Niamey Average Temperature Monthly')
    # axs[1].plot(predictions2, label= 'prediction')
    # axs[1].plot(cities[1].iloc[len(cities[1]) - 24:], label='Actual')
    # axs[1].set_title('Kuwait, Kuwait Average Temperature Monthly')
    # axs[1].legend()
    # axs[2].plot(predictions3, label= 'prediction')
    # axs[2].plot(cities[2].iloc[len(cities[2]) - 24:], label='Actual')
    # axs[2].set_title('UAE, Dubai Average Temperature Monthly')
    # fig.tight_layout()
    # plt.show()