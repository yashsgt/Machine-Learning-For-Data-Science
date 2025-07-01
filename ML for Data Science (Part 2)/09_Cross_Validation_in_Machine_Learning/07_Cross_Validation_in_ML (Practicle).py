import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/placement (2).csv")
print(df.head(3))

x = df.iloc[:,:-1]
y = df['package']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold, StratifiedKFold

p = cross_val_score(LinearRegression(), x, y, cv=5)     # In cv you can take any no. or can provide the different methods Cross Validation
print(p)

print("----------------------------------------------------------------------------------------------------------------------------------------------")

d = cross_val_score(LinearRegression(), x, y, cv=KFold(n_splits = 10))   
print(d)

print("----------------------------------------------------------------------------------------------------------------------------------------------")

f = cross_val_score(LinearRegression(), x, y, cv=LeavePOut(p = 2))   
print(f)

print("----------------------------------------------------------------------------------------------------------------------------------------------")

new_data = df.head(10)

x_new = new_data.iloc[:,:-1]
y_new = new_data['package']

from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold, StratifiedKFold

'''[Using LeaveOneOut method]'''
lo = LeaveOneOut()

for train, test in lo.split(x_new, y_new):
    print(train, test)   # In leave one out method you can see that starting 9 data are in training and left 1 data is in testing or validation
    
print("-----------------------------------------------------------------------------------------------------------------------------------------------")

'''[Using LeavePOut CV method]'''
lp = LeavePOut(p=2)   # Here 'p' is a parameter which decides pair of how many data you want in validation or testing and left data will be for training 

for train, test in lp.split(x_new, y_new):
    print(train, test)
    
print("-----------------------------------------------------------------------------------------------------------------------------------------------")

'''[Using K-Fold CV method]'''
kf = KFold(n_splits=5)   # Here 'n_split' is a parameter which decides how many part you want your data to split

for train, test in kf.split(x_new, y_new):
    print(train, test)
    
print("-----------------------------------------------------------------------------------------------------------------------------------------------")

'''[Using Stratified K-Fold CV method]'''    # This method only works in classification analysis