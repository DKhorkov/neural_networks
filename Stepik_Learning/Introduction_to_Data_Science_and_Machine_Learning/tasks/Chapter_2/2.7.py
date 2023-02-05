import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt


"""
Скачайте набор данных с тремя переменными: sex, exang, num. Представьте, что при помощи дерева решений мы хотим 
классифицировать есть или нет у пациента заболевание сердца (переменная num), основываясь на двух признаках: 
пол (sex) и наличие/отсутствие стенокардии (exang). Обучите дерево решений на этих данных, используйте entropy в 
качестве критерия.
 
Укажите, чему будет равняться значение Information Gain для переменной,  которая будет помещена в корень дерева.

В ответе необходимо указать число с точностью 3 знака после запятой.
"""
data = pd.read_csv('train_data_tree.csv')
features = data.drop(['num'], axis=1)
result = data['num']
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(features, result)
tree.plot_tree(
    clf,
    feature_names=list(features),
    class_names=['No num', 'Num'],
    filled=True
)
plt.show()

# Ответом будет 0.996 - 157/238*0.903 - 81/238*0.826 = 0.119
