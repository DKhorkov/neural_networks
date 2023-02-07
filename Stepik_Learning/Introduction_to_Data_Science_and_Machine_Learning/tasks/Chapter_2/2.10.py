import pandas as pd

data = pd.read_csv('submissions_data_train.csv')

wrong_submissions = data[data.submission_status == 'wrong']

# Если группировать только по user_id или step_id - получим в итоге неправильный степ:
grouped = wrong_submissions.groupby(['user_id', 'step_id'], as_index=False).aggregate({'timestamp': 'max'}).\
    rename(columns={'timestamp': 'last_timestamp'})

last_step = wrong_submissions.merge(grouped, how='outer')
print(last_step)

processed = last_step[last_step.timestamp == last_step.last_timestamp]
print(processed)

counted = processed.groupby('step_id').aggregate({'user_id': "count"}).rename(columns={'user_id': 'counted'})\
    .sort_values(by='counted', ascending=False).iloc[0]
print(counted)
