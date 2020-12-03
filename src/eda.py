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
    return pd.to_datetime(df[['Year', 'Month']].assign(day=1))

def drop_month_year(df):
    '''
    Creates a dataframe that 
    '''
    df.index = df.Date
    return df.drop(['Year', 'Month', 'Date'], axis = 1)

# def city_ts(df):
#     df['Date'] = df_create(df, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
#     df = datetime_month(df)
#     return drop_month_year(df)

city_temp = pd.read_csv('data/cleaned_city_temps.csv')
city_temp = city_temp.drop('Unnamed: 0', axis = 1)
city_temp = city_temp.loc[city_temp['Year'] > 2004]


#Created global_temp as a basic EDA for some cool graphs
#As well as do a Seasonal Decomposition for it
global_temp = df_create(city_temp, ['Region', 'Country', 'City', 'Day'], 
                        ['Year', 'Month'])
global_temp['Date'] = datetime_month(global_temp)
global_temp = drop_month_year(global_temp)
seasonal_result = seasonal_decompose(global_temp, model='additive')



city_highest = df_create(city_temp, ['Region',], 
                                    ['Country', 'City', 'Year', 'Month', 'Day'])
city_highest = city_highest.loc[city_highest['Year'] == 2019 ]

                                

# Creating the 5 cities I'm going to model
niamey = city_temp.loc[city_temp['City'] == 'Niamey']
kuwait = city_temp.loc[city_temp['City'] == 'Kuwait']
dubai = city_temp.loc[city_temp['City'] == 'Dubai']
doha = city_temp.loc[city_temp['City'] == 'Doha']
chennai = city_temp.loc[city_temp['City'] == 'Chennai (Madras)']

niamey = df_create(niamey, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
niamey['Date'] = datetime_month(niamey)
niamey = drop_month_year(niamey)

kuwait = df_create(kuwait, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
kuwait['Date'] = datetime_month(kuwait)
kuwait = drop_month_year(kuwait)

dubai = df_create(dubai, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
dubai['Date'] = datetime_month(dubai)
dubai = drop_month_year(dubai)

doha = df_create(doha, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
doha['Date'] = datetime_month(doha)
doha = drop_month_year(doha)

chennai = df_create(chennai, ['Region', 'Country', 'City', 'Day'], ['Year', 'Month'])
chennai['Date'] = datetime_month(chennai)
chennai = drop_month_year(chennai)


## ARIMA PORTION
result = adfuller(niamey)
df_log = np.log(niamey)
df_log_ma = df_log.rolling(2).mean()
df_detrend = df_log - df_log_ma
df_detrend.dropna(inplace=True)

result = adfuller(df_detrend)

df_log_diff = df_log.diff(periods=2).dropna()
df_diff_rolling = df_log_diff.rolling(12)
df_diff_std = df_diff_rolling.std()

result = adfuller(df_log_diff)

## Using AutoArima but it looks like a diff of 2 periods with a log of 1 will be best
# stepwise_fit = auto_arima(niamey['AvgTemperature'], start_p = 1, start_q = 1, max_p = 5, max_q = 5, m = 12,
#                           start_P = 0, seasonal = True, d = None, D=1, trace = True, error_action = 'ignore',
#                           suppress_warning = True, stepwise = True)

from statsmodels.tsa.statespace.sarimax import SARIMAX 

train = niamey.iloc[:len(niamey)- 24]
test = niamey.iloc[len(niamey) - 24:]

model = SARIMAX(train,
                order = (1, 0 ,3),
                seasonal_order =(4, 2, 1, 12) )
result = model.fit()


if __name__ == '__main__':
    
    ## Creating seasonal dceomposition graph
    # fig, axs = plt.subplots(4, figsize = (14, 12))
    # plot_seasonal_decomposition(axs, global_temp, seasonal_result)
    # fig.tight_layout()
    # plt.show()
    
    # city_highest.head(1).plot.bar(color = 'b',figsize =(14,8))
    # plt.show()
    # print(niamey.head(50))
    # print(niamey.head())
    # print(kuwait.head())
    # print(dubai.head())
    # print(doha.head())
    fig, ax = plt.subplots(figsize=(12, 8))

    start = len(train)
    end = len(train) + len(test) - 1

    predictions = result.predict(start, end, typ='levels').rename('Predictions')

    ax.plot(predictions, label= 'prediction')
    ax.plot(test, label='Actual')
    ax.legend()
    plt.show()