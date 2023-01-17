import pandas as pd

data = pd.read_csv('titanic.csv')

shape = data.shape
print(shape)

data_types = data.dtypes
print(data_types)
