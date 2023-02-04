import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn
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

# Поиграемся с критериями, влияющими на качество нашей модели:
train_and_test_scores = pd.DataFrame(columns=['max_depth', 'train_score', 'test_score'])
max_depth_values = range(1, 101)

for max_depth in max_depth_values:
    classifier_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    classifier_tree.fit(train_features, train_result)
    train_score = classifier_tree.score(train_features, train_result)
    test_score = classifier_tree.score(test_features, test_result)

    temp_dataframe = pd.DataFrame({'max_depth': [max_depth], 'train_score': [train_score], 'test_score': [test_score]})
    train_and_test_scores = pd.concat([train_and_test_scores, temp_dataframe])

print(train_and_test_scores, train_and_test_scores.isnull().sum())

"""Отрисуем наш датафрейм, чтобы наглядно увидеть лучшее значение глубины дерева для точности модели.
Для удобства отрисовки, объединим с помощью пандаса значения тестовых и тренировочных данных в одну колонку, а также
сделаем для них группировку."""
train_and_test_scores_long = pd.melt(
    train_and_test_scores,
    id_vars=['max_depth'],  # основной индекс
    value_vars=['train_score', 'test_score'],  # столбцы для объединения
    var_name='set_type',  # название столбца классификации
    value_name='score'  # название столба со значениями
)
plot = seaborn.lineplot(
    data=train_and_test_scores_long,
    x=train_and_test_scores_long.max_depth,
    y=train_and_test_scores_long.score,
    hue=train_and_test_scores_long.set_type
)
plot.set_xticks(range(1, 101))
plt.xticks(rotation=-90)
plt.show()


"""Однако наши модели все еще переобучены, ведь мы используем один и тот же набор данных для их тренировки и 
тестирования. 
Для решения данной проблемы необходимо разделить набор данных как и ранее, однако тренировочный набор 
следует тоже разделить, например, на 5 мини-тренировочных наборов данных. Допустим, 1 набор данных будет выступать 
тестовым. Тогда мы обучим модель на 2, 3, 4 и 5 наборах, а потом тестируем на том самом 1 наборе данных. Так мы делаем
для каждого мини-наборе данных, чтобы каждый из наборов был и в обучении, и в тесте. А далее, например, можно усреднить 
точность модели всех 5 случаев. А только потом мы будем скармливать моделям тестовые данные.

Такой процесс называется кросс-валидацией."""

# Проверим вышесказанное на модели, с глубиной дерева равной 3:
classifier_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
cross_score = cross_val_score(classifier_tree, train_features, train_result, cv=5)  # cv=5 - делим на 5 наборов данных

# Точность, которую показал классификатор. Сначала обучился на 4, протестил 5. Потом на 1-3, 5 и показал 4. И так далее:
print(cross_score)
average_cross_score = cross_score.mean()
print(average_cross_score)

# Теперь с этими знаниями попробуем снова провести эксперимент с глубиной дерева:
new_train_and_test_scores = pd.DataFrame(columns=['max_depth', 'train_score', 'test_score'])
new_max_depth_values = range(1, 101)

for max_depth in new_max_depth_values:
    classifier_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    classifier_tree.fit(train_features, train_result)
    train_score = classifier_tree.score(train_features, train_result)
    test_score = classifier_tree.score(test_features, test_result)

    average_cross_score = cross_val_score(classifier_tree, train_features, train_result, cv=5).mean()

    temp_dataframe = pd.DataFrame({'max_depth': [max_depth],
                                   'train_score': [train_score],
                                   'test_score': [test_score],
                                   'avg_cross_val_score': [average_cross_score]})
    new_train_and_test_scores = pd.concat([new_train_and_test_scores, temp_dataframe])

print(new_train_and_test_scores, new_train_and_test_scores.isnull().sum())

# Отрисуем новые значения моделей:
new_train_and_test_scores_long = pd.melt(
    new_train_and_test_scores,
    id_vars=['max_depth'],
    value_vars=['train_score', 'test_score', 'avg_cross_val_score'],
    var_name='set_type',
    value_name='score'
)
new_plot = seaborn.lineplot(
    data=new_train_and_test_scores_long,
    x=new_train_and_test_scores_long.max_depth,
    y=new_train_and_test_scores_long.score,
    hue=new_train_and_test_scores_long.set_type
)
new_plot.set_xticks(range(1, 101))
plt.xticks(rotation=-90)
plt.show()

"""Видим, что на самом деле наилучшая точность при кросс-валидации. 
Также стоит отметить, что данные мешаются каждый раз с новым зерном выборке для кросс-валидации."""
check = new_train_and_test_scores_long.query('set_type=="avg_cross_val_score"').\
    sort_values(by=['score'], ascending=False).head(10)
print(check)

# Получим динамический лучший классификатор для теста на валидационных (test_features, test_result) данных:
best_max_depth = check['max_depth'].iloc[0]
print(best_max_depth)
best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=best_max_depth)
best_avg_cross_val_test_data = cross_val_score(best_clf, test_features, test_result, cv=5).mean()
print(best_avg_cross_val_test_data)
