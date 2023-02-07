import pandas as pd
import matplotlib.pyplot as plt

user_data = pd.read_csv('processed_data.csv')
submission_data = pd.read_csv('submissions_data_train.csv')
submission_data['date'] = pd.to_datetime(submission_data.timestamp, unit='s')
submission_data['day'] = submission_data['date'].dt.date

event_data = pd.read_csv('event_data_train.csv')
event_data['date'] = pd.to_datetime(event_data.timestamp, unit='s')
event_data['day'] = event_data['date'].dt.date

pd.set_option('display.max_columns', None)  # Сброс ограничений на число столбцов
print(user_data.head())

"""Посмотрим распределение юзеров, прошедших курс, по количеству уникальных дней на курсе.
На основе данных возьмем промежуток в 3 дня для цели сохранить юзера на курсе:"""
plot = user_data.query('course_passed == True').unique_days_on_course.hist()
plt.show()
threshold = 3 * 24 * 60 * 60

"""Узнаем timestamp начало каждым юзером курса и добавим в сводную и начальную таблички.
Далее отфильтруем только те данные, которые находятся в промежутке трех дней с момента регистрации"""
user_start_timestamp = event_data.groupby('user_id', as_index=False).aggregate({'timestamp': 'min'}). \
    rename(columns={'timestamp': 'start_course_timestamp'})
user_data = user_data.merge(user_start_timestamp, how='outer')
event_data = event_data.assign(start_course_timestamp=event_data.groupby(['user_id'])['timestamp'].transform('min'))

trial_period_data = event_data[event_data['timestamp'] < event_data['start_course_timestamp'] + threshold]
print(trial_period_data)

# Выведем пользователей, которые прошли курс или бросили его:
gone_or_passed_users = user_data[user_data.course_passed | user_data.user_is_gone]


features = gone_or_passed_users.drop(['course_passed', 'user_is_gone'], axis=1)
print(features)
result = gone_or_passed_users.course_passed.map(int)
print(result)
