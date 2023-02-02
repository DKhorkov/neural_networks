import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus
import matplotlib.pyplot as plt


"""Обучить модель, чтобюы она на основе фич предсказала, выживет ли пассажир"""

titanic = pd.read_csv('titanic.csv')
print(titanic, titanic.columns)

nullable_values = titanic.isnull().sum()  # Узнаем, сколько пропущенных данных по каждой колонке датафрейма
print(nullable_values)

"""Отбросим PassengerId, Name, Ticket, Cabin поскольку они не информативны для обучения, 
а также Survived - целевую переменную. Так мы получим наши фичи"""
features = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)  # axis=1 - удаляем колонки
result = titanic.Survived

"""Поскольку DecisionTreeClassifier может работать только с числовыми данными, необходимо преобразовать наши строковые 
значения в числовой эквивалент. Pandas.get_dummies() получает на вход датафрейм и разделяет строковые переменные на 
множество числовых переменных например sex поделит на sex_<Значение>, то есть sex_male и sex_female. Однако, может
быть дублирование, поэтому можно, например, удалить один из них, ведь они эквивалентны"""
features = pd.get_dummies(features)

"""Также дерево решений не умеет работать с NaN значениями. В идеале лучше обрабатывать данные, например, обучить 
микро-модель. Или медиану по полу. но рассчитаем самым простым способом"""
features = features.fillna({'Age': features.Age.median()})
print(features, features.columns)

classifier_tree = tree.DecisionTreeClassifier(criterion='entropy')
classifier_tree.fit(features, result)

# Способ записи картинки модели в пдф:
export = tree.export_graphviz(
    classifier_tree, out_file=None, feature_names=list(features), class_names=['Died', 'Survived'], filled=True
)
graph = pydotplus.graph_from_dot_data(export)
graph.write_pdf('titanic.pdf')

"""Получаемый результат так себе для человека. Таким данные получены, поскольку мы сказали нашей модели не
вычленить закономерность для выводов, а дали датафрейм и результат для цели создания такого дерева решения, которое 
способно безошибочно классифицировать все наблюдения в данном датафрейме на основе Энтропии. По сути, произошел  
over-featuring или переобучение из-за 100% классификации.

В реальных задачах классификации не всегда требуется 100% точности, поскольку порой это всегда возможно.
Таким образом, переобученная модель пригодна только для определенного набора данных, а для прогнозов нам необходимо
не переобучить модель, а вычленить паттерн, на основе которого мы будем делать наш прогноз.

Самый простой и безболезненный способ - ограничить глубину дерева. 
Также следует давать модели 2 набора данных: для обучения и тестирования."""

# Разделим наши данные на обучающие и для тестирования:
train_features, test_features, train_result, test_result = train_test_split(features,
                                                                            result,
                                                                            test_size=0.33,  # Процент на тест
                                                                            random_state=42)  # Зерно выборки

"""Убедимся, что наше дерево решений слишком сильно пытается обучиться на базовом наборе данных. Видим падение. 
Scores - % of correct answers"""
train_scores = classifier_tree.fit(train_features, train_result).score(train_features, train_result)
test_scores = classifier_tree.score(test_features, test_result)
print(train_scores, test_scores)

"""Ограничим глубину дерева, чтобы оно дальше не тратило все силы на полную классификацию, а искало самый лучший 
паттерн для оценки и прогноза данных. При уменьшении глубины до 3, несмотря на уменьшение базового процента корректности 
классификации, происходит наименьшее снижение процента данной классификации на тестовых данных."""
classifier_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
train_scores = classifier_tree.fit(train_features, train_result).score(train_features, train_result)
test_scores = classifier_tree.score(test_features, test_result)
print(train_scores, test_scores)
