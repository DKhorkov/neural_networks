import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/sanyathisside/Predicting-Heart-Disease-using-Machine-" \
      "Learning/master/heart-disease.csv"

df = pd.read_csv(url)
x_train = df.loc[:, "age":"thal"]
y_train = df["target"]

np.random.seed(0)

rf = RandomForestClassifier(10, max_depth=5)
rf.fit(x_train, y_train)

imp = pd.DataFrame(rf.feature_importances_, index=x_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
plt.show()
