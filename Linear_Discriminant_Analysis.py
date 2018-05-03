import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data_frame = pd.read_csv('diabetes.csv')

array = data_frame.values

class_column = array[:, 7]
remaining_values = array[:, 0:7]

num_of_splits = 10
seed = 7

kfold = KFold(n_splits=num_of_splits, random_state=seed)
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, remaining_values, class_column, cv=kfold)
results_mean_value = results.mean()
print(results_mean_value)
