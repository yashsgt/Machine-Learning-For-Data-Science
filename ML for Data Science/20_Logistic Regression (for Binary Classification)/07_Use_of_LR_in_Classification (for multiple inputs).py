import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv("C:/Users/Yash/Downloads/placement (3).csv")

print(df.head(4))

print("------------------------------------------------------------------------------------------------------------------------------------------------")

plt.figure(figsize=(5, 3))
sns.scatterplot(x = 'cgpa', y = 'resume_score', data = df, hue='placed')
plt.show()  # Here you can see it is a linearly separable data

print("------------------------------------------------------------------------------------------------------------------------------------------------")
x = df.iloc[:,:-1]
y = df['placed']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)*100)

print("------------------------------------------------------------------------------------------------------------------------------------------------")
print(lr.predict([[8.14, 6.52]]))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.to_numpy(), y.to_numpy(), clf=lr)
plt.show()

print(lr.coef_)    # It will give you two coefficients beacuse we have two inputs
print("------------------------------------------------------------------------------------------------------------------------------------------------")
print(lr.intercept_)

#  Here algorithm used is:->   y = 1/(1 + e^-x)   eqn of line where x is y' in which y' = m1x1 + m2x2 + c    