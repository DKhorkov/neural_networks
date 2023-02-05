import pandas as pd


end = '\n\n'

dota = pd.read_csv('dota_hero_stats.csv')
pupa_and_lupa = pd.read_csv('accountancy.csv')
algae = pd.read_csv('algae.csv')

"""Let's recalculate the number of legs of Dota2 heroes! Group the heroes from the dataset by the number of their legs 
(legs column), and fill in their number in the task below"""
data_by_legs = dota.groupby('legs').aggregate({'legs': 'count'})
print(data_by_legs, end=end)

"""We received data from the accounting department about the earnings of Lupa and Pupa for different tasks! Look at 
which of them has the highest average earnings in various categories (Type column) and fill in the table, indicating 
the performer with the highest earnings in each of the categories."""
pupa_and_lupa_result = pupa_and_lupa.groupby(['Executor', 'Type'], as_index=False).aggregate({'Salary': 'mean'})
print(pupa_and_lupa_result, end=end)

"""Let's continue our exploration of Dota2 heroes. Group by the attack_type and primary_attr columns and choose the 
most common set of characteristics."""
most_common_set = dota.groupby(['attack_type', 'primary_attr']).aggregate({'primary_attr': 'count'})
print(most_common_set, end=end)

"""Using the previous data, indicate, separated by a space (without commas), what are the minimum, average and maximum 
concentrations of alanine (alanin) among species of the genus Fucus. Round to the 2nd decimal place, decimal point is 
the decimal point."""
alanin_data = algae.groupby(['genus']).aggregate({'alanin': ['min', 'mean', 'max']})
print(alanin_data, end=end)

"""Group algae by group variable and match questions with answers"""
algae_by_broup = algae.groupby(['group'])
print(algae_by_broup.head(11), end=end)
citrate = algae_by_broup.aggregate({'citrate': 'var'})
print(citrate, end=end)
count = algae_by_broup.count()
print(count, end=end)
