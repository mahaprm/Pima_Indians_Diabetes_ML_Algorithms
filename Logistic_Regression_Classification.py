import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

data_frame = pd.read_csv("diabetes.csv")

array = data_frame.values

class_column_as_Y = array[:, 7]
remaining_columns_as_X = array[:, 0:7]

no_of_splits = 10
seed = 7
kfold = KFold(n_splits=no_of_splits, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, remaining_columns_as_X, class_column_as_Y, cv=kfold)
print(results.mean())
