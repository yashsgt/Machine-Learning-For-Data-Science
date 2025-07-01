import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Yash/Downloads/Position_Salaries.csv")

# Polynomial Regression is a regression algorithm that models the relationship between a dependent (y) and independent (x) as
# nth degree polynomial

# Polynomial regression is used when there is no linear relation between input and output
# Y = B* + B1X1 + B2(X1)2 + B3(X1)3 + ....... + Bn(X1)n

print(df)
print("----------------------------------------------------------------------------------------------------------------")

plt.scatter(df['Level'], df['Salary'])
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()  

print("----------------------------------------------------------------------------------------------------------------")
df.drop(columns=['Position'], inplace=True)
print(df)

print("----------------------------------------------------------------------------------------------------------------")
print(df.corr())   # They are highly correlated linearly

print("----------------------------------------------------------------------------------------------------------------")
x = df.iloc[:,:-1]
print(x)
y = df['Salary']
print(y)

print("----------------------------------------------------------------------------------------------------------------")
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=5)
pf.fit(x)
x = pf.transform(x)
print(x)

print("----------------------------------------------------------------------------------------------------------------")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)*100)   # score in percentage
print("----------------------------------------------------------------------------------------------------------------")
'''Drawing the prediction line on the graph'''
pred = lr.predict(x)
plt.scatter(df['Level'], df['Salary'])
plt.plot(df['Level'], pred, c = 'red')
plt.xlabel('level')
plt.ylabel('Salary')
plt.legend(["original", "pred"])
plt.show()  

# Equation of curve:    Y = m1x1 + m2x2^2 + c

print(lr.coef_)   # It will give the value of m1 and m2

print(lr.intercept_)   # It will give the value of y-intercept

# For multiple turned line in prediction line there is 5 slopes m1, m2, m3, and m4, m5 so,  y = m1x1 + m2x2^2 + m3x3^2 +m4x4^2 m5x5^5 + c




'''Checking the accuracy of ML model'''

test = pf.transform([[5]])
# print(test)
print(lr.predict(test)) # It is giving Salary = 110882.13726 for Level = 5 with an accuracy of 99.5%