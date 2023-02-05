import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

"""Продолжим улучшать нашу модель из прошлых уроков на основе данных о выживших на титанике.
Но вместо того, что мы сделали в уроке 2.4 руками - сделаем с помощью GridSearchCV"""

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

decision_tree = tree.DecisionTreeClassifier()
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}

"""Берем классификатор, параметры и настройки кросс-валидации и для каждой комбинации параметров прогоняем обучение на 
кросс-валидации равной "cv" и выбрать такие параметры, для которой будет лучшее значение "accuracy" на 
кросс-валидационном множестве"""
grid_searchCV_clf = GridSearchCV(estimator=decision_tree, param_grid=params, cv=5)
grid_searchCV_clf.fit(train_features, train_result)
best_grid_params = grid_searchCV_clf.best_params_
print(best_grid_params)

# Также можно найти лучшее дерево решений:
best_clf = grid_searchCV_clf.best_estimator_
best_clf_score = best_clf.score(test_features, test_result)  # accuracy лучшего дерева
print(best_clf_score)

# Сделаем prediction лучшего классификатора и предскажем precision_score для него на тестовых данных:
prediction = best_clf.predict(test_features)
precision = precision_score(test_result, prediction)
print(precision)

# Узнаем также значение recall:
recall = recall_score(test_result, prediction)
print(recall)

"""Узнать вероятность отнесения пассажиров к классу умерших или выживших можно следующим образов.
По умолчанию классификация по классу происходит в зависимости от того, больше значение вероятности чем 0.5 или нет"""
prediction_probabilities = best_clf.predict_proba(test_features)  # умрут, выживут
# print(prediction_probabilities)

# Рассмотрим распределение людей и их шансы на выживание:
survived = prediction_probabilities[:, 1]  # Все строки, но только колонка с выжившими
histogram = pd.Series(survived).hist()  # Гистограмма, где 1 - выживание. Если поменять [:, 1] на [:, 0], то 1 - смерть
plt.show()

"""Допустим, мы хотим увеличить значение точности (precision) и считать выжившими только тех, у кого вероятность выжить 
больше 0.8, а не стандартных 0.5.
Для этого перекодируем вероятности в 1 - выжить. если больше 0.8 и остальное в 0 -смерть"""
new_prediction_data = np.where(survived > 0.8, 1, 0)
# print(new_prediction_data)

new_precision = precision_score(test_result, new_prediction_data)
new_recall = recall_score(test_result, new_prediction_data)
print('\n\n', new_precision, new_recall)


"""Нарисуем ROC-кривую. Также известна как кривая ошибок. Чем ближе она к пунктиру - тем хуже модель"""
fpr, tpr, thresholds = roc_curve(test_result, survived)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate == Wrong answer')  # Как часто для умерших мы говорили, что они выжили
plt.ylabel('True Positive Rate == Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

