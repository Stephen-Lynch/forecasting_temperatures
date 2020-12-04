import pandas as pd 
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
city_temp = pd.read_csv('city_temperature.csv', low_memory = False)
city_temp = city_temp.drop(['State'], axis = 1).loc[city_temp['AvgTemperature'] > -90]
city_temp = city_temp.loc[city_temp['Year'] > 2004]





if __name__ == '__main__':
    print(check_for_nulls(city_temp, city_temp.columns))
    city_temp.to_csv('cleaned_city_temps.csv')