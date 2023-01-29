import pandas as pd
import seaborn
import matplotlib.pyplot as plt

events_data = pd.read_csv('event_data_train.csv')
events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')  # Переводим в datetime, unit в секундах

# Стандартные примеры работы с встроенными в pandas методами работы с датами:
events_data['day'] = events_data['date'].dt.date
events_data['year'] = events_data['date'].dt.year
events_data['second'] = events_data['date'].dt.second

# Группируем данные по дате, где значением будут уникальные user_id для даты:
grouped_by_day = events_data.groupby('day').nunique().user_id

# Отрисуем сгруппированные данные:
seaborn.set(rc={'figure.figsize': (9, 6)})  # Настройка для изменения размера графика.
grouped_by_day.plot()
plt.show()

# Неправильная и правильная агрегация данных. Неправильная ведет к потере тех юзеров, которые не решили ни одного степа:
incorrect = events_data[events_data.action == 'passed'].groupby('user_id', as_index=False). \
    aggregate({'step_id': 'count'}).rename(columns={'step_id': 'passed_steps'})

"""Создаем таблицу, в которой будет подсчет step_id для каждого action per user"""
correct = events_data.pivot_table(index='user_id', columns='action', values='step_id', aggfunc='count', fill_value=0)
correct = correct.reset_index()  # Уберем наложение индексов друг на друга

# Посмотрим на количество уникальных юзеров с целью демонстрации важности отбора данных для дальнейшего использования:
print(f'Original data unique user_id: {events_data.user_id.nunique()}',
      f'Incorrect data unique user_id: {incorrect.user_id.nunique()}',
      f'Correct data unique user_id: {correct.user_id.nunique()}')

plt.plot(
    ['Original', 'Incorrect', 'Correct'],
    [events_data.user_id.nunique(), incorrect.user_id.nunique(), correct.user_id.nunique()]
)
plt.show()
