import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv(r"C:/Users/Yash/Downloads/placement (2).csv")

print(df.head(3))

x = df.iloc[:,:-1]
y = df['package']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.score(x_train, y_train)*100, lr.score(x_test, y_test)*100)

dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
print(dtr.score(x_train, y_train)*100, dtr.score(x_test, y_test)*100)

sv = SVR()
sv.fit(x_train, y_train)
print(sv.score(x_train, y_train)*100, sv.score(x_test, y_test)*100)

print("\n")
print("=====================================================================================================================================================")
print("\n")

'''Now making the best model by the combination of all used models'''
from sklearn.ensemble import VotingRegressor     # Voting regressor gives the answer by calculating the average of scores of the all used models 
li = [('lr1',LinearRegression()), ('dt1',DecisionTreeRegressor()), ('sv1',SVR())]
vr = VotingRegressor(li, weights=[10, 8, 9])
vr.fit(x_train, y_train)
print(vr.score(x_train, y_train)*100, vr.score(x_test, y_test)*100)


'''Now Let's see how Voting Regressor(gives the best model) works by using the combination of models'''

pred = {"lr":lr.predict(x_test), "dtr":dtr.predict(x_test), "sv":sv.predict(x_test), "vr":vr.predict(x_test)}
df2 = pd.DataFrame(pred)
print(df2)

