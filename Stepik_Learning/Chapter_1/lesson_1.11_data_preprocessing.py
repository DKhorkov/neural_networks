import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

events_data = pd.read_csv('event_data_train.csv')
events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
events_data['day'] = events_data['date'].dt.date

submissions_data = pd.read_csv('submissions_data_train.csv')
submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
submissions_data['day'] = submissions_data['date'].dt.date

# Рассчитаем для каждого юзера информацию о событиях, совершенных юзером на курсе:
user_events = events_data.pivot_table(index='user_id',
                                      columns='action',
                                      values='step_id',
                                      aggfunc='count',
                                      fill_value=0).reset_index()
print(user_events)

# Рассчитаем для каждого юзера, сколько у него было correct submission_status:
submission_status = submissions_data.pivot_table(index='user_id',
                                                 columns='submission_status',
                                                 values='step_id',
                                                 aggfunc='count',
                                                 fill_value=0).reset_index()
print(submission_status)

"""Рассчитаем перерыв в днях, от которого мы будем отталкиваться в классификации того, дропнул ли юзер курс или пока 
в процессе"""
drop_data = events_data[['user_id', 'day', 'timestamp']]

# Уберем дубликаты user_id и day, а timestamp возьмем любой, чтобы увидеть активность юзера по дням:
drop_data = drop_data.drop_duplicates(subset=['user_id', 'day'])

# Соберем для каждого юзера timestamps, в которые он был на сайте в формате списка:
users_with_timestamps = drop_data.groupby('user_id')['timestamp'].apply(list)

# Узнаем, какой была разница в посещениях курса дял каждого пользователя между его timestamps:
diff_between_timestamps = users_with_timestamps.apply(np.diff)

# Объединим все в одну структуру данных:
diff_array_of_arrays = diff_between_timestamps.values

# Полученный массив из массивов объединим в единый массив для дальнейшей работы:
diff_array = np.concatenate(diff_array_of_arrays, axis=0)  # axis=0, ибо нам нужно объединить строки
diff_dataframe = pd.DataFrame(diff_array, columns=['timestamp_diff'])  # Для удобства переведем данные в dataframe
diff_dataframe['in_days'] = diff_dataframe['timestamp_diff'] / (24 * 60 * 60)
print(diff_dataframe)

# Создадим пандовскую серию и посмотрим статистику, через сколько дней возвращались юзеры к курсу:
gap_data = pd.Series(diff_dataframe['in_days'])
gap_data[gap_data < 200].hist()
plt.show()

# Узнаем, через сколько дней возвращалось 90 и 95 процентов юзеров:
ninty_percents = gap_data.quantile(0.9)
ninty_five_percents = gap_data.quantile(0.95)
print(ninty_percents, ninty_five_percents)

# Узнаем, какой был последний день в нашей выборке:
data_upload_day = events_data['day'].iloc[-1]

# Соберем датафрейм, где для каждого юзера будет его последний день посещения курса:
last_day_visit = events_data.groupby('user_id', as_index=False).aggregate({'timestamp': 'max'})
last_day_visit['date'] = pd.to_datetime(last_day_visit.timestamp, unit='s').dt.date

# Определим, сколько дней прошло с последнего посещения пользователем курса до текущего дня:
last_day_visit['days_gone'] = data_upload_day - last_day_visit['date']

# Если последнее посещение было позже, чем значение нашего трешхолда, то юзер дропнул курс:
drop_out_threshold = pd.to_timedelta(30)
last_day_visit['user_is_gone'] = last_day_visit['days_gone'] > drop_out_threshold

"""Добавим к получившейся таблице данные о количестве успешных и проваленных попыток, а также добавим информацию 
о событиях, совершенных юзером.
on='user_id' - столбец, по которому будет мердж. По умолчанию - по столбцу с одинаковым называнием в каждой таблице.
how='outer' - чтобы не потерять ни одного значения из каждой таблицы."""
merged_data = last_day_visit.merge(submission_status, on='user_id', how='outer')
merged_data['correct'] = merged_data['correct'].fillna(0)
merged_data['wrong'] = merged_data['wrong'].fillna(0)
merged_data = merged_data.merge(user_events, on='user_id', how='outer')
print(merged_data)

# Посчитаем, сколько уникальных дней пользователь посещал курс:
unique_days_by_user = events_data.groupby('user_id').day.nunique().to_frame().reset_index().\
    rename(columns={'day': 'unique_days_on_course'})
merged_data = merged_data.merge(unique_days_by_user, on='user_id', how='outer')
print(merged_data)

# Проверка, что ничего не потеряно:
assert events_data.user_id.nunique() == merged_data.user_id.nunique()

#  Добавим колонку, которая будет отвечать, прошел ли юзер курс или нет, в зависимости от пройденный заданий:
merged_data['course_passed'] = merged_data['passed'] > 170
print(merged_data)

# Посмотрим группировку, сколько юзеров прошло курс, а сколько дропнуло + в процессе:
grouped_by_pass = merged_data.groupby('course_passed').agg('count').reset_index()
print(grouped_by_pass)

# !!! https://stepik.org/lesson/222126/step/8?unit=195047  - Типы  join-ов !!!

# Запишем полученные данные в csv-файл:
merged_data.to_csv('processed_data.csv')
