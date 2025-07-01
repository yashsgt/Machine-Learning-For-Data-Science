import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Yash/Downloads/Iris.csv")
print(df)
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print(df.head())
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print(df['Species'].unique())   # It will give the unique or all different Species type available in Species column
print(df['Species'].nunique())   # It will give the no. of unique or all different Species type available in Species column

print("--------------------------------------------------------------------------------------------------------------------------------------------")
plt.figure(figsize=(5, 3))
sns.pairplot(data=df, hue='Species')
plt.show()

x = df.iloc[:,:-1]
y = df["Species"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

'''Applying OVR method of Logistic Regression'''
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='ovr')
lr.fit(x_train, y_train)
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print(lr.score(x_test, y_test)*100)
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print(lr.predict([[1, 5.1, 3.5, 1.4, 0.2]]))
