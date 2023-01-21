import pandas as pd


end = '\n\n'

data = pd.read_csv('../StudentsPerformance.csv')

# 1) What proportion of students from the dataset have free/reduced in the lunch column?:
total_num = data.shape[0]
num_of_free_reduce = data[data.lunch == 'free/reduced'].shape[0]
print(num_of_free_reduce / total_num, end=end)

# 2) How do the mean and variance of grades in subjects differ between groups of students with a standard/reduced lunch?
free_reduce_lunch = data[data.lunch == 'free/reduced']
standard_lunch = data[data.lunch == 'standard']
print(free_reduce_lunch.describe(), end=end)
print(standard_lunch.describe(), end=end)

print(free_reduce_lunch.iloc[:, -1:-4:-1].var(), end=end)
print(standard_lunch.iloc[:, -1:-4:-1].var(), end=end)
