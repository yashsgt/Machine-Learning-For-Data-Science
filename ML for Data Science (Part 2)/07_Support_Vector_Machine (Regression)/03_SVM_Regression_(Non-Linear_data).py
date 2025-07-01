import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv("C:/Users/Yash/Downloads/Position_Salaries.csv")
print(df)

print(df.isnull().sum())

print(df.describe())

print(df.drop(columns=['Position'], inplace = True))

print(df.head(3))

plt.figure(figsize=(5, 3))
sns.scatterplot(x = 'Level', y = 'Salary', data=df)
plt.show()

x = df[['Level']]
y = df['Salary']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.svm import SVR
svr = SVR(kernel='linear')
svr.fit(x_train, y_train)

print(svr.score(x_test, y_test)*100)
print(svr.score(x_train, y_train)*100)


sns.scatterplot(x = 'Level', y = 'Salary', data=df)
plt.plot(df['Level'], svr.predict(x), color = 'red')
plt.show()