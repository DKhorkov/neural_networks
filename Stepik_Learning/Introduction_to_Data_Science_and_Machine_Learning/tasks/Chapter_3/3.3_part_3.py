import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


"""Thanks to your efforts, the invaders were defeated, but the war is not over yet! You have been promoted and 
transferred to a new direction (new positions - new tasks) - now you need to identify dangerous regions of space 
where buggers can be.

Analyze the available data on the danger of different regions of space and indicate the most likely causes of the 
threat.

Using RandomizedSearchCV instead of GridSearchCV to speed up estimating"""
data = pd.read_csv('space_can_be_a_dangerous_place.csv')
features = data.drop(['dangerous'], axis=1)
result = data['dangerous']

params = {'n_estimators': range(10, 21),
          'max_depth': range(1, 5),
          'min_samples_leaf': range(1, 10),
          'min_samples_split': range(2, 10)
          }

forest = RandomForestClassifier()
grid = RandomizedSearchCV(estimator=forest, param_distributions=params, cv=3, n_jobs=-1)
grid.fit(features, result)

best_clf = grid.best_estimator_
features_importances = best_clf.feature_importances_
feature_importances_dataframe = \
    pd.DataFrame({'features': list(features), 'feature_importances': features_importances}).\
    sort_values('feature_importances', ascending=False)
print(feature_importances_dataframe)
