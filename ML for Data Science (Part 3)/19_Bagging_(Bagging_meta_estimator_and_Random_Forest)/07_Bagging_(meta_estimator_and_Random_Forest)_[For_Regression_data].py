import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=1000, noise=0.2)

data = {"x1":x[:,0], "x2":x[:,1], "y":y}
df = pd.DataFrame(data)
print(df)

sns.scatterplot(x = 'x1', y = 'x2', data=df, hue='y')
plt.show()

x_a = df.iloc[:,:-1]
y_a = df['y']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_a, y_a, test_size=0.2, random_state=42)

'''Using Bagging Algorithm'''
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

br = BaggingRegressor(estimator=SVR(), n_estimators=40)
br.fit(x_train, y_train)
print(br.score(x_train, y_train)*100, br.score(x_test, y_test)*100)

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_train, y_train)
print(rfr.score(x_train, y_train)*100, rfr.score(x_test, y_test)*100)

