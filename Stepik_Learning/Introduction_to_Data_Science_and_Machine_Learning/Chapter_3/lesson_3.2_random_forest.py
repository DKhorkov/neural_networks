import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


# Распределим, как и ранее, данные на тестовые и тренировочные:
titanic = pd.read_csv('titanic.csv')
features = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)
result = titanic.Survived
features = pd.get_dummies(features)
features = features.fillna({'Age': features.Age.median()})

train_features, test_features, train_result, test_result = train_test_split(features,
                                                                            result,
                                                                            test_size=0.33,
                                                                            random_state=42)

"""Упростим параметры, опустив min_samples_split и min_samples_leaf, + добавилось кол-во деревьев в RandomForest.
Также видим, что логика программы не сильно меняется. Поменялся лишь классификатор"""
params = {'n_estimators': [10, 20, 30], 'criterion': ['gini', 'entropy'], 'max_depth': [2, 5, 7, 10]}

forest = RandomForestClassifier()
grid_search_tv = GridSearchCV(estimator=forest, param_grid=params, cv=5)
grid_search_tv.fit(train_features, train_result)

best_params = grid_search_tv.best_params_
print(best_params)

best_clf = grid_search_tv.best_estimator_
print(best_clf.score(test_features, test_result))

# Также в нашем лучшем классификаторе можно увидеть взвешенные метрики для каждой фичи:
feature_importances = best_clf.feature_importances_
feature_importances_dataframe = \
    pd.DataFrame({'features': list(test_features), 'feature_importances': feature_importances}).\
    sort_values('feature_importances', ascending=False)
print(feature_importances_dataframe)
