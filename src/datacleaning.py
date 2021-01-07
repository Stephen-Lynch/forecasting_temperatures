import pandas as pd 
import numpy as np
pd.options.mode.chained_assignment = None


def check_for_nulls(df, columns):
    '''
    Checks for null values inside of a dataset and will tell you how many nulls
    are in each column.

    ARGS:
        df - dataframe
        columns - column labels
    RETURN:
        Dictionary with columns listed that have null values if there are any.
    '''

    true_values = {}
    for col in columns:
        true_values[col] = df[col].isnull().sum()
    return true_values

#Creating initial dataset from which every other dataset will come from
# city_temp = pd.read_csv('city_temperature.csv', low_memory = False)
# city_temp = city_temp.drop(['State'], axis = 1).loc[city_temp['AvgTemperature'] > -90]
# city_temp = city_temp.loc[city_temp['Year'] > 2004]
city_temp = pd.read_csv('data/city_temperature.csv', low_memory = False)
city_temp_USA = city_temp[city_temp['Country'] == 'US']
city_temp_col = city_temp_USA[city_temp['State'] == 'Colorado']
city_temp_is_99 = city_temp_col[city_temp_col['AvgTemperature'] == -99]
city_temp_above_99 = city_temp_col[city_temp_col['AvgTemperature'] > -99]
avgTemp = np.mean(city_temp_above_99['AvgTemperature'])
city_temp_col['AvgTemperature'] = city_temp_col['AvgTemperature'].replace(-99, avgTemp)




if __name__ == '__main__':
    # print(check_for_nulls(city_temp, city_temp.columns))
    city_temp_col.to_csv('col_city_temps.csv')
