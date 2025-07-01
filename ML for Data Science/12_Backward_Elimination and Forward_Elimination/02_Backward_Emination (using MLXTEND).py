'''Backward Elimination Technique:-> Backward elimination is like tidying up a model. You start with all the features
  (variables) included, and then remove one feature at a time. After removing a feature, you check if the model's
  performance gets worse. If it doesn't, you leave that feature out. You keep doing this until you can't remove any
  more features without harming the model's performance.'''

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
df = pd.read_csv("C:/Users/Yash/Downloads/diabetes.csv")
print(df)

print(df.shape)

print("-------------------------------------------------------------------------------------------------------")
print(df.head(3))

print("-------------------------------------------------------------------------------------------------------")
x = df.iloc[:,:-1]
y = df['Outcome']

from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

print("--------------------------------------------------------------------------------------------------------")
fs = SequentialFeatureSelector(lr, k_features = 4, forward = False)   # You can select any no. of features and do feaures selection (forward = False) for backward elimination
fs.fit(x, y)

# Print selected feature indices
print("Selected feature indices: ", fs.k_feature_idx_)

# Print selected features name
selected_features = [x.columns[idx] for idx in fs.k_feature_idx_]
print("Selected Features name: ", selected_features)

# Printing the score of the selected features
print("Score: ", fs.k_score_)

