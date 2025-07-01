'''Feature Selection Techniques:-> A feature is an attribute that has an impact on a problem or is useful for the problem,
   and choosing the important feature for the model is known as feature selection
   
   In other words, we can say feature selection means selecting those relevant things which are important for machine learning
   models and the things can be rows, columns, values, etc
   
   and for feature selection you must have the knowledge of the domain'''
   

   
'''Forward elimination Technique:-> Forward elimination is like building a model step by step. You start with nothing and
   then keep adding one feature (or variable) at a time to your model. Each time you add a feature, you check if it improves
   the model's performance. If it does, you keep it; if not, you leave it out. You stop when adding more features doesn't make
   the model any better.'''
   
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
fs = SequentialFeatureSelector(lr, k_features = 5, forward = True)   # You can select any no. of features and do feaures selection, and here (forward = True) for forward elimination
fs.fit(x, y)

# Print selected feature indices
print("Selected feature indices: ", fs.k_feature_idx_)

# Print selected features name
selected_features = [x.columns[idx] for idx in fs.k_feature_idx_]
print("Selected Features name: ", selected_features)

# Printing the score of the selected features
print("Score :", fs.k_score_*100)

