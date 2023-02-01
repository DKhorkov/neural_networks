import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'X_1': [1, 1, 1, 0, 0, 0, 0, 1],
    'X_2': [0, 0, 0, 1, 0, 0, 0, 1],
    'Y': [1, 1, 1, 1, 0, 0, 0, 0]
})

"""Задача предсказать значение и изменение Y в зависимости от значений X_1, X_2 с помощью дерева решений. 
Энтропия - значение, в зависимости от которого можно сделать вывод о том, к какому классу отнести событие. Чем оно ниже, 
тем больше регрессия на основе входных данных. 

По сути, Энтропия (Е) - уровень неопределенности нашей модели https://habr.com/ru/company/ods/blog/322534/
Выбор порядка использования фич основывается на показателе прироста информации IG"""
our_tree_classifier = tree.DecisionTreeClassifier(criterion='entropy')  # Создали дерево решений с критерием энтропии

"""Сделаем предобработку данных. Все данные (фичи) сохраним в одну переменную, а целевые значение( ожидаемый результат) 
в другую"""
features = data[['X_1', 'X_2']]  # Dataframe
target_values = data['Y']  # Series

# Обучим наше дерево решений:
our_tree_classifier.fit(features, target_values)

# Нарисуем результат обучения дерева:
tree.plot_tree(
    our_tree_classifier,
    feature_names=list(features),
    class_names=['Negative', 'Positive'],  # Лейблы для удобства восприятия (y=1 - positive, y=0 - negative)
    filled=True  # Раскраска для наглядности
)
plt.show()
