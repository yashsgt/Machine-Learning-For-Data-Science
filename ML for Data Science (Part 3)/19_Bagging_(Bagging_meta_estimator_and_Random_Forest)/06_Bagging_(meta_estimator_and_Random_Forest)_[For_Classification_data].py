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
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

bc = BaggingClassifier(estimator=SVC(), n_estimators=50)   # In Bagging meta estimator default algorithm is Decision Tree but you change give any other algorithm
bc.fit(x_train, y_train)
print(bc.score(x_train, y_train)*100, bc.score(x_test, y_test)*100)

print("==============================================================================================================================================")

rfc = RandomForestClassifier(n_estimators=10)     # In random forest the default algorithm is Decision Tree always, you can't change the algorithm in Random Forest
rfc.fit(x_train, y_train)
print(rfc.score(x_train, y_train)*100, rfc.score(x_test, y_test)*100)

