'''Filing the categorical nan value with its mode value'''

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (1).csv")
print(df)

# print(df['country'].fillna(df['country'].mode()[0]))
print("-----------------------------------------------------------------------------------------")

print(df['country'].mode()[0])  # with mode()[0] it will give the first most repeating data

print("-----------------------------------------------------------------------------------------")
# Filling the missing value with mode value for categorical data of the dataset
df['country'].fillna(df['country'].mode()[0], inplace = True)
print(df)
null_count = df['country'].isnull().sum()
print(f"The null values in 'country' is = {null_count}")   # Here we are getting zero null value because we have filled country's null value with its mode value 

'''When we have to fill mode value to our all categorical or object data type we will use for loop'''
for i in df.select_dtypes(include='object').columns:
    (df[i].fillna(df[i].mode()[0], inplace = True))
    
print("-----------------------------------------------------------------------------------------")
print(df.isnull().sum())  # Here, we can see in all categorical data {country, continent, day, time} ther is no null value left becuse they all are filled by the mode value of their corresponding columns

    
