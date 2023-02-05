import matplotlib.pyplot as plt
import pandas as pd
import seaborn


"""You are given a dataset with 2 features (columns). Plot the distribution of points (observations) in the space of 
these 2 variables (one of them will be x and the other - y) and write the number of clusters formed by the observations.
In your answer, you must specify the number of clusters as a number (for example: 3)."""
data = pd.read_csv('dataset_209770_6.txt', sep=' ')
plot = seaborn.lmplot(x='x', y='y', data=data, fit_reg=False)

"""Download data representing the genomic distances between species and build a heatmap to visualize the differences. 
In response, enter which picture corresponds to the downloaded data."""
geom_matrix = pd.read_csv('genome_matrix.csv').set_index('Unnamed: 0')  # Меняем индекс, чтобы нарисовать карту
geom_matrix_plot = seaborn.heatmap(data=geom_matrix, cmap="viridis", annot=True)
geom_matrix_plot.xaxis.set_ticks_position('top')
geom_matrix_plot.xaxis.set_tick_params(rotation=90)

"""It's time to find out who is the most important cancer and what role in dota is the most common. Download the 
dataset with data about the heroes from the dota 2 game and look at the distribution of their possible roles in the 
game (roles column). Build a bar chart showing how many heroes how many roles are assigned (according to Valve, of 
course) and write how many roles most heroes have."""
dota = pd.read_csv('dota_hero_stats.csv')
roles_length = [len(r.split(',')) for r in dota.roles]
dota['roles_length'] = roles_length
roles = dota.groupby('roles_length', as_index=False).aggregate({'id': 'count'})
print(roles)

"""Now let's move on to the flowers. Graduate student Adele decided to study what irises are. Help Adele learn more 
about irises - download the dataset with iris parameter values, plot their distributions and mark the correct 
statements by looking at the graph."""
flowers = pd.read_csv('iris.csv')
for column in flowers:
    seaborn.displot(flowers, x=column, kde=True)

"""Consider the length of the petals (petal length) in more detail and use the violin raft for this. Draw the length 
distribution of the irises petals from the previous dataset using the violin plot and select the correct (same) option 
from the suggested ones."""
violin = seaborn.violinplot(data=flowers['petal length'])

"""We continue to study irises! Another important type of graphs is pairplot, which shows the dependence of pairs of 
variables on each other, as well as the distribution of each of the variables. Build it and look at the scatter plots 
for each feature pair. Which of the pairs offhand has the highest correlation?"""
corr = seaborn.pairplot(flowers, hue="species")

plt.show()
