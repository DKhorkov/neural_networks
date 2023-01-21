import pandas as pd


end = '\n\n'

data = pd.read_csv('StudentsPerformance.csv')

print(data.head(), end=end)  # Начало таблицы
print(data.describe(), end=end)  # Описательная статистика данных
print(data.dtypes, end=end)  # Типы переменных

# Сгруппировать данные таблицы по гендеру и вывести медиану письменных очков для данных гендеров:
print(data.groupby('gender').aggregate({'writing score': 'mean'}), end=end)

# Вывести первые пять строк и первые три столбца для них:
print(data[:5]['gender'], end=end)
print(data.iloc[:5, :5], end=end)  # по индексу
print(data.loc[[0, 1, 2, 3, 4], ['gender', 'writing score']], end=end)  # по индексу и столбцам

# Замена индексов на буквы:
data_with_indices = data.iloc[0:5]
data_with_indices.index = ['a', 'b', 'c', 'd', 'e']
print(data_with_indices, end=end)

# Датафреймы и серии:
print(type(data_with_indices), end=end)  # DataFrame - то есть набор из серий под каждый столбец/лейбл
sorted_data_with_indices = data_with_indices.iloc[:, 1]
print(sorted_data_with_indices, end=end)
print(type(sorted_data_with_indices), end=end)  # Series

# Создание серии (то есть массива с именами/лейблами):
pandas_series_1 = pd.Series(data=[1, 2, 3], index=['one', 'two', 'three'])
print(pandas_series_1, end=end)
pandas_series_2 = pd.Series(data=['a', 'b', 'c'], index=['one', 'two', 'three'])
print(pandas_series_1, end=end)

# Создание датафрейма:
pandas_dataframe = pd.DataFrame({'Numbers': pandas_series_1, 'Letters': pandas_series_2})
print(pandas_dataframe, end=end)

# Разница в синтаксисе для вывода датафрейма и серии из него:
print(pandas_dataframe[['Numbers']], end=end)  # Датафрейм, где будет только колонка гендера
print(pandas_dataframe['Numbers'], end=end)  # Серия с гендером из датафрейма
