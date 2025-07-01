import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv("C:/Users/Yash/Downloads/placement (2).csv")
print(df.head())

print(df.isnull().sum())

plt.figure(figsize=(5, 3))
sns.scatterplot(x = 'cgpa', y = 'package', data=df)
plt.show()

x = df[['cgpa']]
y = df['package']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.svm import SVR
svr = SVR(kernel='linear')
svr.fit(x_train, y_train)

print(svr.score(x_test, y_test)*100)
print(svr.score(x_train, y_train)*100)

from sklearn.tree import DecisionTreeRegressor
dtc = DecisionTreeRegressor()
dtc.fit(x_train, y_train)

for i in range(1, 21):
    dtc = DecisionTreeRegressor(max_depth=i)
    dtc.fit(x_train, y_train)
    print(i, dtc.score(x_train, y_train)*100, dtc.score(x_test, y_test)*100)
    
'''Drawing Regression Line'''
sns.scatterplot(x = 'cgpa', y = 'package', data=df)
plt.plot(df['cgpa'], svr.predict(x), color = 'red')
plt.show()
    
    
    

