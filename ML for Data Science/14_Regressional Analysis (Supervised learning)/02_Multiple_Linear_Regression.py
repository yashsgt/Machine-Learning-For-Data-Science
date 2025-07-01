'''Multiple Linear regression is an extension of Simple linear regression as it takes more than one predictor variable to
   predict the response variable'''
   
'''Multiple Linear Regression is a statistical method used to predict the value of one variable (called the dependent variable)
   based on the values of two or more other variables (called independent variables). Essentially, it helps to find a straight-
   line relationship between the dependent variable and multiple independent variables.'''
   
'''That means suppose we have two independent varibles X1 and X2 and a dependent variable Y. Then for each independent variable
   there must be a linear relation with dependent variable ,  if this condition satisfies then we can apply Multiple linear
   regression'''
   
'''Y = M1X1 + M2X2 + M3X3 + ...MnXn + C'''  # This is the general equation for Multiple Linear regression 

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt  

df = pd.read_csv("C:/Users/Yash/Downloads/Salary_Data.csv")
print(df)  
print("-----------------------------------------------------------------------------------------------------------")

print(df.head(3))

print("-----------------------------------------------------------------------------------------------------------")
print(df.shape)

print("-----------------------------------------------------------------------------------------------------------")
print(df.isnull().sum())  # Here you can see that there is no null values

print("-----------------------------------------------------------------------------------------------------------")
plt.figure(figsize=(10, 7))
sns.pairplot(df)
            # sns.pairplot() is a function from the Seaborn library in Python. It creates a grid of scatter plots for all
            # numerical columns in a DataFrame, showing pairwise relationships between them. This is particularly useful for
            # visualizing correlations and distributions in your dataset.
plt.show()   # Here you can that the graph is correlated linearly so multiple linear regression can be applied

print("-----------------------------------------------------------------------------------------------------------")
'''Drawing the heatmap'''
sns.heatmap(data = df.corr(), annot = True)  # the parameter annot stands for annotation. When set to True, it displays the actual data values (correlation coefficients, in this case) on the heatmap. These values are shown as text within the cells of the heatmap, making it easier to interpret the data visually.
                                             # If you set annot = False (or it, as False is the default), the heatmap will not display these values.

plt.show()

print("-----------------------------------------------------------------------------------------------------------")
x = df.iloc[:,:-1]   # left the last column 'Salary'
print(x.ndim)   # Checking the dimension of input data (it should be 2 for Linear Regression)
print(x) 

print("-----------------------------------------------------------------------------------------------------------")
y = df['Salary']
print(y)

print("-----------------------------------------------------------------------------------------------------------")
'''Training and testing of the data'''
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)  # testing size is 20% of the data
print(x_train)
print(x_test)

print(y_train)
print(y_test)

print("-----------------------------------------------------------------------------------------------------------")
from sklearn.linear_model import LinearRegression
Lr = LinearRegression()
Lr.fit(x_train, y_train)  # input data is already in 2-D so need not to use double[]

'''Checking the score or accuracy of machine learning model'''
print(Lr.score(x_test, y_test))    # It gives us the accuracy of ML model in percentage
print("-----------------------------------------------------------------------------------------------------------")
print(Lr.predict([[1.1, 21.0]]))  # It will give value near 39343 (see in 1st row)

print("-----------------------------------------------------------------------------------------------------------")       
# For y = m1x1 + m2x2 + .... + C                       
print(Lr.coef_)   # It will give the value for m1 for x1  and m2 for x2
print("-----------------------------------------------------------------------------------------------------------")
print(Lr.intercept_)   # It will give the value for c
print("-----------------------------------------------------------------------------------------------------------")
print(x.columns)     # Printing the total no. of columns in x
print("-----------------------------------------------------------------------------------------------------------")

# y_pred = 4882.14850701*YearExperience + 2567.51865301*Age  +  -20612.69192148531





