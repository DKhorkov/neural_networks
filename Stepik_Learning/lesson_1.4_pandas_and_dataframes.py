import pandas as pd
import numpy as np

data = pd.read_csv('StudentsPerformance.csv')

print(data.head(), end='\n\n')  # Начало таблицы
print(data.describe(), end='\n\n')  # Описательная статистика данных
print(data.dtypes, end='\n\n')  # Типы переменных

# Сгруппировать данные таблицы по гендеру и вывести медиану письменных очков для данных гендеров:
print(data.groupby('gender').aggregate({'writing score': 'mean'}), end='\n\n')

# Вывести первые пять строк и первые три столбца для них:
print(data[:5]['gender'], end='\n\n')
print(data.iloc[:5, :5], end='\n\n')  # по индексу
print(data.loc[[0, 1, 2, 3, 4], ['gender', 'writing score']], end='\n\n')  # по индексу и столбцам

# Замена индексов на буквы:
data_with_indices = data.iloc[0:5]
data_with_indices.index = ['a', 'b', 'c', 'd', 'e']
print(data_with_indices, end='\n\n')

# Датафреймы и серии:
print(type(data_with_indices), end='\n\n')  # DataFrame - то есть набор из серий под каждый столбец/лейбл
sorted_data_with_indices = data_with_indices.iloc[:, 1]
print(sorted_data_with_indices, end='\n\n')
print(type(sorted_data_with_indices), end='\n\n')  # Series

# Создание серии (то есть массива с именами/лейблами):
pandas_series_1 = pd.Series(data=[1, 2, 3], index=['one', 'two', 'three'])
print(pandas_series_1, end='\n\n')
pandas_series_2 = pd.Series(data=['a', 'b', 'c'], index=['one', 'two', 'three'])
print(pandas_series_1, end='\n\n')

# Создание датафрейма:
pandas_dataframe = pd.DataFrame({'Numbers': pandas_series_1, 'Letters': pandas_series_2})
print(pandas_dataframe, end='\n\n')

# Разница в синтаксисе для вывода датафрейма и серии из него:
print(pandas_dataframe[['Numbers']], end='\n\n')  # Датафрейм, где будет только колонка гендера
print(pandas_dataframe['Numbers'], end='\n\n')  # Серия с гендером из датафрейма
