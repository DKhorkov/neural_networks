import seaborn
import matplotlib.pyplot as plt
import pandas as pd

end = '\n\n'

data = pd.read_csv('StudentsPerformance.csv')

# Встроенная в pandas визуализация данных из matplotlib:
math_score_picture = data['math score'].hist()
data.plot.scatter(x='math score', y='reading score')

# Отрисуем данные с помощью seaborn (настройки matplotlib):
plot = seaborn.lmplot(x='math score', y='reading score', hue='gender', data=data)
plot.set_xlabels('Math scores')  # Меняем названяи осей
plot.set_ylabels('Reading scores')
plt.show()
