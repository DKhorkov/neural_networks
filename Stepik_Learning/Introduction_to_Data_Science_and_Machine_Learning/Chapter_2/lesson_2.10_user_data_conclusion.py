import pandas as pd
import matplotlib.pyplot as plt

user_data = pd.read_csv('processed_data.csv')
event_data = pd.read_csv('event_data_train.csv')
event_data['date'] = pd.to_datetime(event_data.timestamp, unit='s')
event_data['day'] = event_data['date'].dt.date

pd.set_option('display.max_columns', None)  # Сброс ограничений на число столбцов
print(user_data.head())

"""Посмотрим распределение юзеров, прошедших курс, по количеству уникальных дней на курсе.
На основе данных возьмем промежуток в 3 дня для цели сохранить юзера на курсе:"""
plot = user_data.query('course_passed == True').unique_days_on_course.hist()
plt.show()

# Узнаем timestamp начало каждым юзером курса и добавим в сводную табличку.
user_start_timestamp = event_data.groupby('user_id', as_index=False).aggregate({'timestamp': 'min'}).\
    rename(columns={'timestamp': 'start_course_timestamp'})
user_data = user_data.merge(user_start_timestamp, how='outer')
print(user_data)
