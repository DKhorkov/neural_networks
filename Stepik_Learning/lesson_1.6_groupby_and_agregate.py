import pandas as pd
import numpy as np


end = '\n\n'

data = pd.read_csv('StudentsPerformance.csv')

"""Сгруппируем датафрейм по гендеру и выведем среднее значение скоров и сумму для каждого гендера. 
as-index=False необходимо, чтобы таблица была более приятна наглядна и гендер не использовался в качестве индекса, 
а был отдельным столбцом:"""
data_by_gender = data.groupby('gender', as_index=False).aggregate({'math score': ['mean', 'sum'],
                                                                   'reading score': ['mean', 'sum'],
                                                                   'writing score': ['mean', 'sum']})
print(data_by_gender, end=end)

# Также можно сгруппировать по нескольким столбцам:
data_by_gender_and_lunch = data.groupby(['gender', 'lunch'], as_index=False).aggregate({'math score': ['mean', 'sum']})
print(data_by_gender_and_lunch, end=end)

# Отберем 5 лучших студентов по гендеру на основе математических оценок:
sorted_by_math_score = data.sort_values(['gender', 'math score'], ascending=False)  # ascending - в порядке возрастания
print(sorted_by_math_score, end=end)
sorted_by_math_score = sorted_by_math_score.loc[:, ['gender', 'math score']]
best_by_math = sorted_by_math_score.groupby('gender', as_index=False).head(5).sort_values('math score', ascending=False)
print(best_by_math, end=end)

# Создадим новый столбец, в котором будет сумма трех столбцов с оценками:
total_score = data.filter(like='score').sum(axis=1)
print(total_score, end=end)
data['total score'] = total_score
print(data, end=end)

# Для создания нескольких новых колонок используем assign. Прологарифмируем total score и math score:
columns = data.columns
new_columns = [name.replace(" ", "_") for name in columns]
data = data.rename(columns=dict(zip(columns, new_columns)))
data = data.assign(total_score_log=np.log(data.total_score), math_score_log=np.log(data.math_score))
print(data, end=end)

# Удалим колонку total_score и math_score с помощью drop. axis=1 - удаляем колонку, а не строку:
new_data = data.drop(['total_score', 'math_score'], axis=1)
print(new_data.columns, end=end)
