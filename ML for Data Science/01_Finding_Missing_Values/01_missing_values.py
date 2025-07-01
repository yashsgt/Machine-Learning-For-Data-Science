import pandas as pd

df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (1).csv")
print(df)

print("----------------------------------------------------------------")
print(df.shape)
print("----------------------------------------------------------------")
print(df.isnull().sum())  # Checking null values for each columns

print("----------------------------------------------------------------")
print(df.isnull().sum().sum())    # Checking total null values in the dataset

print("----------------------------------------------------------------")
print((df.isnull().sum())/(df.shape[0])*100)   # Checking the percentage of null values for each columns


print("----------------------------------------------------------------")
print((df.isnull().sum().sum())/(df.shape[0]*df.shape[1])*100)    # Checking the percentage of total null values in the dataset


print("----------------------------------------------------------------")
print(df.notnull().sum().sum())   # Checking the total no. of not null values in the dataset

