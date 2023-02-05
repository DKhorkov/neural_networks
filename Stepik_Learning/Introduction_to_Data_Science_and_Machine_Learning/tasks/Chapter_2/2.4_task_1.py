import pandas as pd
from sklearn import tree
import seaborn
import matplotlib.pyplot as plt


"""Убедимся в том, что всё так происходит на самом деле. Скачайте тренировочный датасэт с ирисами, обучите деревья с 
глубиной от 1 до 100. Целевой переменной при обучении является переменная species. При этом записывайте его скор 
(DecisionTreeClassifier.score()) на тренировочных данных, и аккуратность предсказаний (accuracy_score) на тестовом 
датасэте. Затем визуализируйте зависимость скора и аккуратности предсказаний от глубины дерева и выберите правильную 
визуализацию из предложенных."""

train_data = pd.read_csv('train_iris.csv', index_col=0)
test_data = pd.read_csv('test_iris.csv', index_col=0)

train_features = train_data.drop(['species'], axis=1)
train_result = train_data.species
test_features = test_data.drop(['species'], axis=1)
test_result = test_data.species


train_and_test_scores = pd.DataFrame(columns=['max_depth', 'train_score', 'test_score'])
max_depth_values = range(1, 100)

for max_depth in max_depth_values:
    classifier_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    classifier_tree.fit(train_features, train_result)
    train_score = classifier_tree.score(train_features, train_result)
    test_score = classifier_tree.score(test_features, test_result)

    temp_dataframe = pd.DataFrame({'max_depth': [max_depth],
                                   'train_score': [train_score],
                                   'test_score': [test_score]})
    train_and_test_scores = pd.concat([train_and_test_scores, temp_dataframe])

train_and_test_scores_long = pd.melt(
    train_and_test_scores,
    id_vars=['max_depth'],
    value_vars=['train_score', 'test_score'],
    var_name='set_type',
    value_name='score'
)
plot = seaborn.lineplot(
    data=train_and_test_scores_long,
    x=train_and_test_scores_long.max_depth,
    y=train_and_test_scores_long.score,
    hue=train_and_test_scores_long.set_type
)
plt.show()
