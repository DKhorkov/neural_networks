import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


transport = {'transport':  0,  'fighter':  1,  'cruiser': 2}

train_data = pd.read_csv('invasion.csv')
train_features = train_data.drop('class', axis=1)
train_result = train_data['class'].map(transport)

params = {'n_estimators': range(10, 21),
          'max_depth': range(1, 5),
          'min_samples_leaf': range(1, 10),
          'min_samples_split': range(2, 10)
          }

forest = RandomForestClassifier()
grid = GridSearchCV(estimator=forest, param_grid=params, cv=3, n_jobs=-1)
grid.fit(train_features, train_result)

best_clf = grid.best_estimator_
best_clf.fit(train_features, train_result)
test_features = pd.read_csv('operative_information.csv')
prediction = best_clf.predict(test_features)
print(pd.Series(prediction).value_counts())


"""Which variable turned out to be the most important for classifying ships?"""
features_importances = best_clf.feature_importances_
feature_importances_dataframe = \
    pd.DataFrame({'features': list(train_features), 'feature_importances': features_importances}).\
    sort_values('feature_importances', ascending=False)
print(feature_importances_dataframe)
