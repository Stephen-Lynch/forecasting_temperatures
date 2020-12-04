import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('ggplot')
import statsmodels.api as sm
import warnings 
warnings.filterwarnings("ignore") 


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



#Created global_temp as a basic EDA tool for some cool graphs
#As well as do a Seasonal Decomposition for it
global_temp = df_create(city_temp, ['Region', 'Country', 'City', 'Day'], 
                        ['Year', 'Month'])
global_temp_year = df_create(city_temp, ['Region', 'Country', 'City', 'Day', 'Month'], ['Year'])
global_temp_year = global_temp_year.loc[global_temp_year['Year'] < 2020]
global_temp_year.index = global_temp_year.Year
global_temp_year = global_temp_year.drop('Year', axis = 1)
global_temp['Date'] = datetime_month(global_temp)
global_temp = drop_month_year(global_temp)
seasonal_result = seasonal_decompose(global_temp, model='additive')

# Creating the 3 cities I'm going to model
niamey = city_temp.loc[(city_temp['City'] == 'Niamey') & (city_temp['Year'] < 2020)]
kuwait = city_temp.loc[(city_temp['City'] == 'Kuwait') & (city_temp['Year'] < 2020)]
dubai = city_temp.loc[(city_temp['City'] == 'Dubai') & (city_temp['Year'] < 2020)]

niamey_year = niamey.drop(['Month', 'Day'], axis = 1).groupby('Year').mean('AvgTemperature')
kuwait_year = kuwait.drop(['Month', 'Day'], axis = 1).groupby('Year').mean('AvgTemperature')
dubai_year = dubai.drop(['Month', 'Day'], axis = 1).groupby('Year').mean('AvgTemperature')

niamey = df_create(niamey, ['Region', 'Country', 'Day', 'City'], ['Year', 'Month',])
niamey['Date'] = datetime_month(niamey)
niamey = drop_month_year(niamey)

kuwait = df_create(kuwait, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
kuwait['Date'] = datetime_month(kuwait)
kuwait = drop_month_year(kuwait)

dubai = df_create(dubai, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
dubai['Date'] = datetime_month(dubai)
dubai = drop_month_year(dubai)





if __name__ == '__main__':
    ## Line graph for Seasonal decomp Global
    #Creating seasonal dceomposition graph
    # fig, axs = plt.subplots(4, figsize = (14, 12), dpi = 100)
    # plot_seasonal_decomposition(axs, global_temp, seasonal_result)
    # fig.tight_layout()
    
    ## Line graph of top 5 hottest cities
    fig, ax = plt.subplots(figsize = (14,8), dpi = 200)
    ax.plot(kuwait_year, label = 'Kuwait, Kuwait')
    ax.plot(niamey_year, label = 'Nigeria, Niamey')
    ax.plot(dubai_year, label = 'UAE, Dubai')
    ax.set_xlabel('Years', fontsize = 20)
    ax.set_ylabel('Temperature in F', fontsize = 20)
    plt.xticks(fontsize= 16)
    plt.yticks(fontsize = 16)
    ax.legend()

    #Line graph for Global temps rising
    # fig, ax = plt.subplots(figsize=(12, 8), dpi = 200)
    # ax.plot(global_temp_year)
    # ax.set_title('Rising temperatures over 10 years', fontsize = 20)
    # ax.set_xlabel('Years', fontsize = 20)
    # ax.set_ylabel('Temperature in F', fontsize = 20)
    # plt.xticks(fontsize = 16)
    
    # plt.xticks(fontsize = 16)
    plt.show()

