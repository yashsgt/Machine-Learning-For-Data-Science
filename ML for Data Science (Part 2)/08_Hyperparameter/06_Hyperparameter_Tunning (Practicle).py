import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/Position_Salaries.csv")

print(df)

print(df.isnull().sum())

print(df.drop(columns=['Position'], inplace = True))

print(df)

x = df.iloc[:,:-1]
y = df['Salary']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

'''Making model'''
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(criterion='squared_error', max_depth = 10, splitter = 'best')
dtr.fit(x_train, y_train)

print(dtr.score(x_train, y_train)*100)
print(dtr.score(x_test, y_test)*100)

for i in range(1, 21):
    dtr = DecisionTreeRegressor(max_depth=i)
    dtr.fit(x_train, y_train)
    print(i, dtr.score(x_train, y_train)*100, dtr.score(x_test, y_test)*100)
    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

'''It is used to provide best parameters to get high accuracy in score'''
df = {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
      "splitter": ["best", "random"],
      "max_depth": [i for i in range(1, 20)]}

print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
'''[Using Grid-Search-CV]'''
gd = GridSearchCV(DecisionTreeRegressor(), param_grid=df)
gd.fit(x_train, y_train)

print(gd.best_params_)

print(gd.best_score_)


print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
'''[Using Randomize-Search-CV]'''
rd = RandomizedSearchCV(DecisionTreeRegressor(), param_distributions=df, n_iter=20)
rd.fit(x_train, y_train)

print(rd.best_params_)

print(rd.best_score_)