import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('ggplot')

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
    axs[0].plot(series.index, series)
    axs[0].set_title("Raw Series")
    axs[1].plot(series.index, sd.trend)
    axs[1].set_title("Trend Component $T_t$")
    axs[2].plot(series.index, sd.seasonal)
    axs[2].set_title("Seasonal Component $S_t$")
    axs[3].plot(series.index, sd.resid)
    axs[3].set_title("Residual Component $R_t$")

def datetime_month(df):
    return pd.to_datetime(global_temp[['Year', 'Month']].assign(day=1))

def drop_month_year(df):
    df.index = df.Date
    return df.drop(['Year', 'Month', 'Date'], axis = 1)

   

city_temp = pd.read_csv('cleaned_city_temps.csv')
city_temp = city_temp.drop('Unnamed: 0', axis = 1)


#Created global_temp as a basic EDA for some cool graphs
global_temp = df_create(city_temp, ['Region', 'Country', 'City', 'Day'], 
                        ['Year', 'Month'])
global_temp['Date'] = datetime_month(global_temp)
global_temp = drop_month_year(global_temp)




seasonal_result = seasonal_decompose(global_temp, model='additive')


if __name__ == '__main__':
    fig, axs = plt.subplots(4, figsize = (14, 12))

    plot_seasonal_decomposition(axs, global_temp, seasonal_result)
    fig.tight_layout()
    plt.show()
    

    print(global_temp.head(5))