import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt


"""Представьте, что вы решили научить робота для животных отличать собачек от котиков. Для начала проанализируем 
данные - отметьте все верные утверждения о данном датафрэйме"""
data = pd.read_csv('dogs.csv')
print(data)

classifier_tree = tree.DecisionTreeClassifier(criterion='entropy')

features = data[['Шерстист', 'Гавкает', 'Лазает по деревьям']]
result = data['Вид']

classifier_tree.fit(features, result)

tree.plot_tree(
    classifier_tree,
    feature_names=list(features),
    class_names=['Dogs', 'Cats'],
    filled=True
)
plt.show()


"""В нашем Big Data датасэте появились новые наблюдения! Давайте немного посчитаем энтропию, чтобы лучше понять, 
формализуемость разделения на группы."""
data = pd.read_csv('cats.csv')

feature_1 = data[['Шерстист']]
feature_2 = data[['Гавкает']]
feature_3 = data[['Лазает по деревьям']]
result = data['Вид']

# Шерстист:
classifier_tree.fit(feature_1, result)

tree.plot_tree(
    classifier_tree,
    feature_names=list(feature_1),
    class_names=['Dogs', 'Cats'],
    filled=True
)
plt.show()

# Гавкает:
classifier_tree.fit(feature_2, result)

tree.plot_tree(
    classifier_tree,
    feature_names=list(feature_2),
    class_names=['Dogs', 'Cats'],
    filled=True
)
plt.show()

# Лазает по деревьям:
classifier_tree.fit(feature_3, result)

tree.plot_tree(
    classifier_tree,
    feature_names=list(feature_3),
    class_names=['Dogs', 'Cats'],
    filled=True
)
plt.show()


"""Ещё немного арифметики - посчитаем Information Gain по данным из предыдущего задания. 
Впишите через пробел округлённые до 2-ого знака значения IG для фичей Шерстист, Гавкает и Лазает по деревьям. 
Десятичным разделителем в данном задании является точка.

Пришлось посчитать лапками: 0.08 0.61 0.97"""