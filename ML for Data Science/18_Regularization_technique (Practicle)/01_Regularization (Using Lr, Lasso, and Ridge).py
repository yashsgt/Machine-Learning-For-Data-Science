import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/Yash/Downloads/houseprice.csv")
print(df)

print("------------------------------------------------------------------------------------------------------------------")

print(df.head(4))
print("------------------------------------------------------------------------------------------------------------------")

sns.heatmap(data=df.corr(), annot=True)
plt.show()

print("------------------------------------------------------------------------------------------------------------------")
# Separating dependent and independent values:
x = df.iloc[:,:-1]
y = df['Price']

'''Scaling of Data'''
sc = StandardScaler()
sc.fit(x)
print(sc.transform(x))
print("------------------------------------------------------------------------------------------------------------------")

''''making dataframe'''
x = pd.DataFrame(sc.transform(x), columns = x.columns)
print(x)

'''Training and Testing of data'''                                                                            
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

print("------------------------------------------------------------------------------------------------------------------")
'''Forming linear model'''


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


'''[LinearRegression]'''
lr = LinearRegression()
lr.fit(x_train, y_train)   # fitting of data

'''Checking the accuracy of ML Model'''
print(lr.score(x_test, y_test)*100)

'''Checking the mean_absolute_error'''
print(mean_absolute_error(y_test, lr.predict(x_test)))
'''Checking the mean_squared_error'''
print(mean_squared_error(y_test, lr.predict(x_test)))
'''Checking the Root_mean_square error'''
print(np.sqrt(mean_squared_error(y_test, lr.predict(x_test))))
print("------------------------------------------------------------------------------------------------------------------")
print(lr.coef_)
print("------------------------------------------------------------------------------------------------------------------")
plt.figure(figsize=(10, 5))
plt.bar(x.columns, lr.coef_)
plt.title('LinearRegression')
plt.xlabel('columns')
plt.ylabel('coef_')
plt.show()
print("------------------------------------------------------------------------------------------------------------------")








'''[Lasso]'''
la = Lasso(alpha=0.5)   
'''In the context of Lasso regression, the alpha parameter is not a "penalty corner" but rather a regularization strength.
   It controls the amount of penalty applied to the coefficients of the regression model. Specifically, Lasso uses an L1
   penalty, which encourages sparsity in the model by shrinking some coefficients to zero. This makes it useful for feature
   selection.
'''
la.fit(x_train, y_train)
# Checking rthe accuracy of model
print(la.score(x_test, y_test)*100)    #  By changing the alpha value you can inc. or dec. the accuracy of ML model


'''Checking the mean_absolute_error'''
print(mean_absolute_error(y_test, la.predict(x_test)))
'''Checking the mean_squared_error'''
print(mean_squared_error(y_test, la.predict(x_test)))
'''Checking the Root_mean_square error'''
print(np.sqrt(mean_squared_error(y_test, la.predict(x_test))))
print("------------------------------------------------------------------------------------------------------------------")
plt.figure(figsize=(10, 5))
plt.bar(x.columns, la.coef_)
plt.title('Lasso')
plt.xlabel('columns')
plt.ylabel('coef_')
plt.show()   # here you can see that by using Lasso, the constant coefficient value is much more reduced (that means reducing +ve and -ve data much more)
print("------------------------------------------------------------------------------------------------------------------")







'''[Ridge]'''
ri = Ridge(alpha=10)
ri.fit(x_train, y_train)

# Checking the accuracy of ML model
print(ri.score(x_test, y_test)*100)   # Here you can see that accuracy of ML model is increased due to Ridge


'''Checking the mean_absolute_error'''
print(mean_absolute_error(y_test, ri.predict(x_test)))
'''Checking the mean_squared_error'''
print(mean_squared_error(y_test, ri.predict(x_test)))
'''Checking the Root_mean_square error'''
print(np.sqrt(mean_squared_error(y_test, ri.predict(x_test))))
print("------------------------------------------------------------------------------------------------------------------")
plt.figure(figsize=(10, 5))
plt.bar(x.columns, ri.coef_)
plt.title('Ridge')
plt.xlabel('columns')
plt.ylabel('coef_')
plt.show()  # here you can see that by using Ridge, the constant coefficient value is much more reduced        
print("------------------------------------------------------------------------------------------------------------------")







'''Comparing the constant coefficients value for Lr, Lasso and Ridge'''
dataset = pd.DataFrame({"col_name":x.columns, "LinearRegression":lr.coef_, "Lasso":la.coef_, "Ridge":ri.coef_})
print(dataset)





