import pandas as pd
from sklearn import tree
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt


# Повторим обработку данных из прошлого урока:
titanic = pd.read_csv('titanic.csv')
features = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)
result = titanic.Survived
features = pd.get_dummies(features)
features = features.fillna({'Age': features.Age.median()})

train_features, test_features, train_result, test_result = train_test_split(features,
                                                                            result,
                                                                            test_size=0.33,
                                                                            random_state=42)

decision_tree = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy')
decision_tree.fit(train_features, train_result)


tree.plot_tree(
    decision_tree,
    feature_names=list(features),
    class_names=['Died', 'Survived'],
    filled=True
)
plt.show()

"""В примере выше дерево все еще переобучено, ведь оно при 95% успешной классификации все равно спускается на 3 уровень, 
где энтропия может только увеличиться.

Добавим еще несколько параметров, которые очень важны и влияют на переобученность нашего дерева.
min_samples_split - если размер выборки меньше заданного параметра, то сплит не будет производиться дальше. 
min_samples_leaf - ограничивает сплит, если размер выборки отсплитованного варианта меньше заданного параметра.
min_impurity_decrease - ожидаемое минимальное уменьшение неопределенности (IG)
"""
decision_tree = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy', min_samples_split=100,
                                            min_samples_leaf=10)
decision_tree.fit(train_features, train_result)


tree.plot_tree(
    decision_tree,
    feature_names=list(features),
    class_names=['Died', 'Survived'],
    filled=True
)
plt.show()


"""А теперь соберем новый GridSearchCV и посмотрим, какие же лучшие значения параметров мы можем взять
Также можно использовать RandomizedSearchCV для ускорения работы и выборочного перебора параметров."""
decision_tree = tree.DecisionTreeClassifier()
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30),
          'min_samples_split': range(10, 200), 'min_samples_leaf': range(1, 50)}
random_searchCV_clf = RandomizedSearchCV(estimator=decision_tree, param_distributions=params, cv=5)
random_searchCV_clf.fit(train_features, train_result)
best_grid_params = random_searchCV_clf.best_params_
print(best_grid_params)
