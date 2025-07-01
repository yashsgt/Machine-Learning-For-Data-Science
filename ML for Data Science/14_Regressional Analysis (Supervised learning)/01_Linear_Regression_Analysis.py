'''Linear regression analysis is a method to understand the relationship between variables by fitting a straight line to the
   data. It helps you analyze how changes in independent variables (the inputs or predictors) influence the dependent variable
   (the outcome or what you're predicting).'''
   
'''Algorithm used :-> Linear Regression algorithm'''

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv("C:/Users/Yash/Downloads/placement.csv")
print(df)
print("---------------------------------------------------------------------------------------------------------------")
'''When input feature is single then, we use Simple Linear Regression'''
'''Simple linear regression is a type of regression algorithms that models the relationship between a dependent variable and
   a single independent variable'''
   
print(df.head(3))
print("---------------------------------------------------------------------------------------------------------------")
plt.figure(figsize=(10, 7))
sns.scatterplot(x = 'cgpa', y = 'package', data=df)
plt.show()  # Here in graph You can see that all dots are towards upwards and one direction (linealy related; 1 output feature with every 1 input)
            # So Here Simple Linear Regression can be followed
print("---------------------------------------------------------------------------------------------------------------")
print(df.isnull().sum())    # Here you can see that no null value is there
print("---------------------------------------------------------------------------------------------------------------")

x = df[['cgpa']]
y = df['package']

print("---------------------------------------------------------------------------------------------------------------")
'''Now training and testing of data'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)   # test size = 20%
# The random_state parameter in train_test_split is used to ensure reproducibility of your data split.
print(x_train)
print(x_test)
print(y_train)
print(y_test)

print("---------------------------------------------------------------------------------------------------------------")

from sklearn.linear_model import LinearRegression 
lr = LinearRegression()
lr.fit(x_train, y_train)  # Training of data, Here fit provide us the value of m and c for [y = mx + c]

print("---------------------------------------------------------------------------------------------------------------")
'''Now checking the accuracy of machine learning model'''
print(lr.score(x_test, y_test))

print(lr.predict([[6.89]]))  # I Have given input cgpa 6.89
                             # It will give the corresponding package near 3.26 that will decide its accuracy according to dataset(see first row)
                             # By changing random_state we can change the accuracy of ML model
print("---------------------------------------------------------------------------------------------------------------")
# for  y = mx + c
print("slope: ", lr.coef_)  # It will give you m

print("y-intercept: ", lr.intercept_)    # It will give you c

print("---------------------------------------------------------------------------------------------------------------")
'''Now Drawing the regressional line or prediction line'''
y_pred = lr.predict(x)
plt.figure(figsize=(10, 7))
sns.scatterplot(x = 'cgpa', y = 'package', data=df, label = 'original data')
plt.plot(df['cgpa'], y_pred, c = 'red', label = 'predict line')    # value of y-axis i.e; y_pred with respect to x-axis cgpa
plt.legend(['original data', 'predict line'])
plt.savefig('predict.jpg')  # Saving graph in form of jpg image
plt.show() 



