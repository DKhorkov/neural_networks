import pandas as pd


"""Найти айди Анатолия Карпова. Логика: больше всего попыток"""

submissions_data = pd.read_csv('../submissions_data_train.csv')
submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
submissions_data['day'] = submissions_data['date'].dt.date

# Рассчитаем для каждого юзера, сколько у него было correct submission_status:
correct_submission_status = submissions_data.pivot_table(index='user_id',
                                                         columns='submission_status',
                                                         values='step_id',
                                                         aggfunc='count',
                                                         fill_value=0).reset_index()

sorted_data = correct_submission_status.sort_values(by='correct', ascending=False)
print(sorted_data)


