import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


"""Let's take heart disease data and train Random Forest on it.
Build a graph of the importance of variables for classification and choose the most similar among the proposed options.

There is randomness in the task, run random forest training and plotting several times to see
changes in the importance of features (the 5 most important are usually in the top, just in a different order).
To get the same graph as in the correct answer, do"""


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
