import pandas as pd
from sklearn import tree
import pydotplus

"""Мы собрали побольше данных о котиках и собачках, и готовы обучить нашего робота их классифицировать! 
Скачайте тренировочный датасэт и  обучите на нём Decision Tree. После этого скачайте датасэт из задания и 
предскажите какие наблюдения к кому относятся. Введите число собачек в вашем датасэте.

В задании допускается определённая погрешность.

P. S.: данные в задании находятся в формате json, используйте метод pd.read_json для их прочтения"""

train_data = pd.read_csv('train_dogs_n_cats.csv')
print(train_data)
test_data = pd.read_json('test_dogs_and_cats.txt')

train_features = train_data.drop(['Вид', 'Длина', 'Высота'], axis=1)
train_result = pd.get_dummies(train_data['Вид']).drop(['собачка'], axis=1)

test_features = test_data.drop(['Длина', 'Высота'], axis=1)
print(test_features)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(train_features, train_result)
export = tree.export_graphviz(
    clf, out_file=None, feature_names=list(train_features), class_names=['Dogs', 'Cats'], filled=True
)
graph = pydotplus.graph_from_dot_data(export)
graph.write_pdf('cats_and_dogs_clf.pdf')

prediction = clf.predict(test_features)
dogs = list(prediction).count(0)
print(prediction, dogs)
