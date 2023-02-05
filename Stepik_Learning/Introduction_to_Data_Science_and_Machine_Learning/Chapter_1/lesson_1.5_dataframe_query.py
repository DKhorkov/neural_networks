import pandas as pd


end = '\n\n'

data = pd.read_csv('StudentsPerformance.csv')

dataframe = data[:20]
print(dataframe, end=end)

# Возврат булева при грубой группировке:
sort_by_gender = dataframe.gender == 'male'
print(sort_by_gender, end=end)

# Фильтрация по loc, принимающему лейблы и индексы:
sort_by_gender_loc = dataframe.loc[dataframe.gender == 'male']
print(sort_by_gender_loc, end=end)

# Отберем те строки, где writing score превышает среднее значение и мужской пол. loc можно не использовать:
average_writing_score = dataframe['writing score'].mean()
print(average_writing_score, end=end)
sort_by_ws_loc = dataframe.loc[(dataframe['writing score'] > average_writing_score) & (dataframe.gender == 'male')]
print(sort_by_ws_loc, end=end)

# Отбор, аналогичный предыдущему, но используя query. Для удобства переименуем лейблы без пробелов:
columns = dataframe.columns
new_columns = [name.replace(" ", "_") for name in columns]
dataframe = dataframe.rename(columns=dict(zip(columns, new_columns)))
query_result = dataframe.query("writing_score > @average_writing_score & gender == 'male'")  # f-strings works too
print(query_result, end=end)

# Отбор колонок, содержащих score в названии:
dataframe_scores_labels = dataframe[[label for label in list(dataframe) if 'score' in label]]  # standard way
print(dataframe_scores_labels, end=end)
dataframe_scores_labels_filtered = dataframe.filter(like='score')  # pandas way
print(dataframe_scores_labels_filtered, end=end)

# Отбор через фильтр строк, а не столбцов:
dataframe_filtered_by_line = dataframe.filter(like='9', axis=0)
print(dataframe_filtered_by_line, end=end)
