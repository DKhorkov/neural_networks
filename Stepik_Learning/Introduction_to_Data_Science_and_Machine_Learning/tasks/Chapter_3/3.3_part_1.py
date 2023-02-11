import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn
import matplotlib.pyplot as plt

"""Iterate over the parameters using GridSearchCV and train Random Forest on the data specified in the previous step. 
Pass a model with random_state to GridSearchCV

RandomForestClassifier(random_state=0)

Options for selection -
     n_estimators: 10 to 50 in steps of 10
     max_depth: 1 to 12 in steps of 2
     min_samples_leaf: 1 to 7
     min_samples_split: 2 to 9 in steps of 2

Specify cv=3. To speed up calculations in GridSearchCV, you can specify n_jobs=-1 to use all processors.
What Random Forest parameters were chosen as the best ones to solve on this data?"""

data = pd.read_csv('training_mush.csv')
train_features = data.drop(['class'], axis=1)
train_result = data['class']

clf = RandomForestClassifier(random_state=0)
params = {'n_estimators': range(10, 51, 10),
          'max_depth': range(1, 13, 2),
          'min_samples_leaf': range(1, 8),
          'min_samples_split': range(2, 10, 2)
          }
grid = GridSearchCV(estimator=clf, param_grid=params, cv=3, n_jobs=-1)
grid.fit(train_features, train_result)
best_params = grid.best_params_
print(best_params)


"""Choose the correct statements about the importance of features for our classifier"""
features_importances = grid.best_estimator_.feature_importances_
feature_importances_dataframe = \
    pd.DataFrame({'features': list(train_features), 'feature_importances': features_importances}).\
    sort_values('feature_importances', ascending=False)
print(feature_importances_dataframe)


"""Now we have a classifier that determines which mushrooms are edible and which are not, let's try it! 
Predict the edibility of these given mushrooms and write back the number of inedible mushrooms (class 1)."""
test_features = pd.read_csv('testing_mush.csv')
best_clf = grid.best_estimator_
best_clf.fit(train_features, train_result)
prediction = best_clf.predict(test_features)
print(np.count_nonzero(prediction))


"""Create a confusion matrix based on the predictions you got in the last lesson and the correct answers 
(use the password from the previous task to open them). Choose the correct one from the given options"""
test_result = pd.read_csv('testing_y_mush.csv')
matrix = confusion_matrix(test_result, prediction)
heatmap = seaborn.heatmap(matrix, cmap=plt.cm.Blues, annot=True,annot_kws={"size": 16})
plt.show()
