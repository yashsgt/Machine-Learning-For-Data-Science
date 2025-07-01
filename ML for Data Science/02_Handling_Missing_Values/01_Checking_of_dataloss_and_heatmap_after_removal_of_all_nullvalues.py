import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (1).csv")
print(df)

print("-------------------------------------------------------------------------------------------------------------------")
print(df.head(5))

print("-------------------------------------------------------------------------------------------------------------------")
print(df.shape)

print("-------------------------------------------------------------------------------------------------------------------")
print(df.isnull().sum())

print("-------------------------------------------------------------------------------------------------------------------")
'''Removing column 'Recovered' which is having most null values'''
df.drop(columns=['Recovered'], inplace=True)

print("-------------------------------------------------------------------------------------------------------------------")
print(df)   # Here you can see the Recovered column is removed

print("-------------------------------------------------------------------------------------------------------------------")
print(df.shape)   # Here you can see that one column is reduced

print("-------------------------------------------------------------------------------------------------------------------")
print(df.dropna(inplace = True))    # Here df.dropna(inplace = True) is used to remove those rows having null values 
print(df.isnull().sum())

# df.dropna(axis = 1)  is used to remove the column having null values
print("____________________________________________________________________________________________________________________")
print(df)

print(df.shape)
print("____________________________________________________________________________________________________________________")
loss = ((238-189)/238) * 100
print(f"We have lost {loss} % of the data from our dataset")

print("____________________________________________________________________________________________________________________")
sns.heatmap(df.isnull())  # Checking null values again after removing all null values using above df.dropna(inplace = True)
plt.show()


