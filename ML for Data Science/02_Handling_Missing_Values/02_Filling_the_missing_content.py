import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (1).csv")
print(df)

print("_____________________________________________________________________________________________________________________")
print(df.head(4))

print("_____________________________________________________________________________________________________________________")
print(df.isnull().sum())

print("_____________________________________________________________________________________________________________________")
(df.fillna(15, inplace=True))   # Filling all the missing value with number 15


print("_____________________________________________________________________________________________________________________")
print(df.isnull().sum())    # You can see no null value left
print(df.head(30))     # Here you can see all NAN values are replaced by 15



print("_____________________________________________________________________________________________________________________")
print(df.info())  # here objects are categorical data
                  # axis = 0  along rows but its effect is observed column-wise i.e, e.g:-> add a number in a column
                  # axis = 1  along columns but its effect is observed row-wise i.e, e.g:-> add a number in a row
                 
print("_____________________________________________________________________________________________________________________")
print(df.fillna(method = 'bfill', axis = 0))  # backward filling along rows but effect shows on column wise 
print(df.fillna(method = 'ffill', axis = 0))  # forward filling along rows but effect shows on column wise 

print("_____________________________________________________________________________________________________________________")
print(df.fillna(method = 'bfill', axis = 1))  # backward filling along columns but effect shows on row wise
print(df.fillna(method = 'ffill', axis = 1))  # forward filling along columns but effect shows on row wise



 