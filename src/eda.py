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

def df_create(df, drop, groupby, reset_index = True):
    '''
    Creates a dataframe based on a past dataframe if you want to dorp or
    groupby certain columns

    ARGS:
        df - dataframe
        drop - list of column labels you would like to drop
        groupby - list of column labels you would like to groupby
        reset_index - Whether you want to keep the index or reset it

    RETURN:
        Dataframe with respective columns dropped and groupby'd
    '''
    temp_df = df.drop(drop, axis= 1).copy()
    temp_df = temp_df.groupby(groupby).mean()
    if reset_index == True:
        return temp_df.reset_index()
    else:
        return temp_df

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
    axs[0].plot(series.index, series)
    axs[0].set_title("Raw Series")
    axs[1].plot(series.index, sd.trend)
    axs[1].set_title("Trend Component $T_t$")
    axs[2].plot(series.index, sd.seasonal)
    axs[2].set_title("Seasonal Component $S_t$")
    axs[3].plot(series.index, sd.resid)
    axs[3].set_title("Residual Component $R_t$")
    

def datetime_month(df):
    '''
    Creates a datetime object consisting of years and months

    Args:
        df - Dataframe

    Returns:
        A series full of monthly date times
    '''
    return pd.to_datetime(df[['Year', 'Month']].assign(day=1))

def drop_month_year(df):
    '''
    Indexes the date and drops the year, month and date column

    Args:
        df - Dataframe

    Returns:
        A new dataframe with it's index in datetime as well as
        column's dropped
    '''
    df.index = df.Date
    return df.drop(['Year', 'Month', 'Date'], axis = 1)

city_temp = pd.read_csv('data/cleaned_city_temps.csv')
city_temp = city_temp.drop('Unnamed: 0', axis = 1)
city_temp_train = city_temp.loc[city_temp['Year'] < 2017]
city_temp_ho = city_temp.loc[(city_temp['Year'] > 2016) 
                            * (city_temp['Year'] < 2019)]


#Created global_temp as a basic EDA tool for some cool graphs
#As well as do a Seasonal Decomposition for it
global_temp = df_create(city_temp_train, ['Region', 'Country', 'City', 'Day'], 
                        ['Year', 'Month'])
global_temp_year = df_create(city_temp_train, ['Region', 'Country', 'City', 'Day', 'Month'], ['Year'], reset_index = False)
global_temp['Date'] = datetime_month(global_temp)
global_temp = drop_month_year(global_temp)
seasonal_result = seasonal_decompose(global_temp, model='additive')

# Creating the 3 cities I'm going to model
niamey = city_temp_train.loc[city_temp_train['City'] == 'Niamey']
kuwait = city_temp_train.loc[city_temp_train['City'] == 'Kuwait']
dubai = city_temp_train.loc[city_temp_train['City'] == 'Dubai']

niamey_year = niamey.drop('Month', axis = 1).groupby('Year').mean('AvgTemperature')
kuwait_year = kuwait.drop('Month', axis = 1).groupby('Year').mean('AvgTemperature')
dubai_year = dubai.drop('Month', axis = 1).groupby('Year').mean('AvgTemperature')

niamey = df_create(niamey, ['Region', 'Country', 'Day', 'City'], ['Year', 'Month',])
niamey['Date'] = datetime_month(niamey)
niamey = drop_month_year(niamey)

kuwait = df_create(kuwait, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
kuwait['Date'] = datetime_month(kuwait)
kuwait = drop_month_year(kuwait)

dubai = df_create(dubai, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
dubai['Date'] = datetime_month(dubai)
dubai = drop_month_year(dubai)

## ARIMA PORTION



# Using AutoArima but it looks like a diff of 2 periods with a log of 1 will be best
# stepwise_fit = auto_arima(dubai['AvgTemperature'], start_p = 1, start_q = 1, max_p = 5, max_q = 5, m = 4,
#                           start_P = 0, seasonal = True, d = None, D=1, trace = True, error_action = 'ignore',
#                           suppress_warning = True, stepwise = True)





if __name__ == '__main__':
    ## Line graph for Seasonal decomp Global
    #Creating seasonal dceomposition graph
    # fig, axs = plt.subplots(4, figsize = (14, 12))
    # plot_seasonal_decomposition(axs, global_temp, seasonal_result)
    # fig.tight_layout()
    
    
    # fig, ax = plt.subplots(figsize=(12, 8))
    # fig, axs = plt.subplots(2, figsize=(20, 4), dpi = 200)
    # start = len(train1)
    # end = len(train1) + len(test1) - 1

    # predictions1 = result1.predict(start, end, typ='levels').rename('Predictions')
    # predictions2 = result2.predict(start, end, typ='levels').rename('Predictions')
    # predictions3 = result3.predict(start, end, typ='levels').rename('Predictions')

    # # Line graph of predictions and tests 
    # axs[0].plot(predictions1, label= 'prediction')
    # axs[0].plot(test1, label='Actual')
    # axs[0].set_title('Nigeria, Niamey Average Temperature Monthly')
    # axs[1].plot(predictions2, label= 'prediction')
    # axs[1].plot(test2, label='Actual')
    # axs[1].set_title('Kuwait, Kuwait Average Temperature Monthly')
    # axs[1].legend()
    # axs[2].plot(predictions3, label= 'prediction')
    # axs[2].plot(test3, label='Actual')
    # axs[2].set_title('UAE, Dubai Average Temperature Monthly')
    # fig.tight_layout()

    
    ## Line graph of top 5 hottest cities
    # ax.plot(kuwait_year, label = 'Kuwait, Kuwait')
    # ax.plot(niamey_year, label = 'Nigeria, Niamey')
    # ax.plot(dubai_year, label = 'UAE, Dubai')
    # ax.set_xlabel('Years', fontsize = 20)
    # ax.set_ylabel('Temperature in F', fontsize = 20)
    # plt.xticks(fontsize= 16)
    # plt.yticks(fontsize = 16)
    # ax.legend()

  

    # fig, axs = plt.subplots((3, figsize=(20, 4))
    # axs[0].plot(niamey.iloc[len(niamey)-24:], label = 'Avg Temp', color = 'b')
    # axs[0].plot(forecast, label ='Forecast', color ='y', linewidth=2)
    # axs[0].legend()

    
    # axs[0].plot(niamey, label = 'Avg Temp', color = 'b')
    # axs[0].plot(forecast1, label ='Forecast', color ='y', linewidth=2)
    # axs[0].legend()
    # axs[0].set_title('Nigeria, Niamey Forecast Average Monthly Temp')
    # axs[1].plot(dubai, label = 'Avg Temp', color = 'b')
    # axs[1].plot(forecast3, label ='Forecast', color ='y', linewidth=2)
    # axs[1].legend()
    # axs[1].set_title('UAE, Dubai Forecast Average Monthly Temp')
    # fig.tight_layout()
    


    
    

    #Line graph for Global temps rising
    # fig, ax = plt.subplots(figsize=(12, 8), dpi = 200)
    # ax.plot(global_temp_year)
    # ax.set_title('Rising temperatures over 10 years', fontsize = 20)
    # ax.set_xlabel('Years', fontsize = 20)
    # ax.set_ylabel('Temperature in F', fontsize = 20)
    
    