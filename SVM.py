import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC

data_frame = pandas.read_csv("diabetes.csv")
array = data_frame.values

class_column_as_Y = array[:, 7]
remaining_columns_as_X = array[:, 0:7]

no_of_splits = 10
seed = 7
kfold = KFold(n_splits=no_of_splits, random_state=seed)
model = SVC()
results = cross_val_score(model, remaining_columns_as_X, class_column_as_Y, cv=kfold)
print(results.mean())
