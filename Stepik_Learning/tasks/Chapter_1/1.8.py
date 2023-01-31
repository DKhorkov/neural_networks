import pandas as pd
import numpy as np


"""In any way convenient for you, create a dataframe with the name my_data, in which there are two columns with the 
names (type - strings, value - integers) and four observations in each column"""
dataframe = pd.DataFrame(data={'type': ['A', 'A', 'B', 'B'], 'value': [10, 14, 12, 23]})


"""In a variable named subset_1 save only the first 10 rows and only columns 1 and 3.
In a variable named subset_2 save all rows except 1 and 5 and only columns 2 and 4.
Remember that row and column index numbers start from 0.
Please note that the resulting subset_1 and subset_2 should also be dataframe"""
my_stat = pd.read_csv('my_stat.csv')
subset_1 = my_stat.iloc[:10, [0, 2]]
subset_2 = my_stat.iloc[lambda x: x.index != (0 or 4), [1, 3]].iloc[1:, :]


"""Now let's practice selecting the observations (rows) we need that correspond to a certain condition. 
The dataframe named my_stat has four columns V1, V2, V3, V4:
In the subset_1 variable, save only those observations whose values of the variable V1 are strictly greater than 0, 
and the value of the variable V3 is equal to 'A'.
In the subset_2 variable, save only those observations whose values of the variable V2 are not equal to 10, or the 
values of the variable V4 are greater than or equal to 1.
As in the previous task, the filtering result is also a dataframe."""
subset_1 = my_stat[(my_stat['V1'] > 0) & (my_stat['V3'] == 'A')]
subset_2 = my_stat[(my_stat['V2'] != 10) | (my_stat['V4'] >= 1)]


"""Now let's transform our data. The my_stat variable contains the data with which you need to do the following action. 
In this data (my_stat) create two new variables:
V5 = V1 + V4
V6 = natural logarithm of variable V2"""
my_stat['V5'] = my_stat['V1'] + my_stat['V4']
my_stat['V6'] = np.log(my_stat['V2'])


"""Great job, fix a couple more important questions and we can move on.
Variables V1, V2 ... such names are no good. With such names, it is easy to get confused in your own data and, as a 
result, make mistakes in the calculations. Rename the columns in my_stat data as follows:
V1 -> session_value, V2 -> group, V3 -> time, V4 -> n_users"""
del my_stat['V5']
del my_stat['V6']
my_stat = my_stat.rename(columns={'V1': 'session_value', 'V2': 'group', 'V3': 'time', 'V4': 'n_users'})


"""And finally, let's figure out how to replace observations in the data.
The dataframe named my_stat has data with 4 columns: session_value, group, time, n_users.
In the session_value variable, replace any missing values with zeros.
In the n_users variable, replace all negative values with the median value of the n_users variable (ignoring negative 
values, of course)."""
my_stat = pd.read_csv('my_stat_V2.csv')
my_stat = my_stat.fillna(0)
median = my_stat[my_stat.n_users >= 0.0].n_users.median()
n_users = list(my_stat.n_users)
for (i) in range(len(n_users)):
    if n_users[i] < 0:
        n_users[i] = median
my_stat['n_users'] = n_users


"""In this task, for the data my_stat, calculate the average value of the session_value variable for each group 
(group variable), in the resulting dataframe, the group variable should not turn into an index. Also rename the 
session_value column to mean_session_value."""
mean_session_value_data = my_stat.groupby('group', as_index=False).aggregate({'session_value': 'mean'}).\
    rename(columns={'session_value': 'mean_session_value'})
