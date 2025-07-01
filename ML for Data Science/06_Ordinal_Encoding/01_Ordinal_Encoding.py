'''Ordinal encoding is a technique used to convert categorical data into numerical data by assigning a unique integer
   to each category, while preserving the natural order or rank of the categories. It is particularly useful for ordinal
   data, where the categories have a meaningful sequence or hierarchy.'''


import pandas as pd
df = pd.DataFrame({"size" : ["s", 'm', 'l', 'xl', 's', 'm', 'l']})
print(df.head(3))

print("--------------------------------------------------------------------------------------------------")
# Deciding order before encoding:->
ord_data = [["s", "m", "l", "xl"]]

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories = ord_data)
print(oe.fit(df[['size']]))

print("--------------------------------------------------------------------------------------------------")
print(oe.transform(df[['size']]))


print("--------------------------------------------------------------------------------------------------")
df['size_en'] = oe.transform(df[['size']])  # Putting encoded data into dataset table
print(df)

print("--------------------------------------------------------------------------------------------------")
'''Use of map-function:-> map function provides the functionality where you can assign any number to a value during ordinal_Encoding'''
ord_data1 = {"s":0, "m":1, "l":2, "xl":3}
df['size_en_map'] = df['size'].map(ord_data1)
print(df)

print("--------------------------------------------------------------------------------------------------")
dataset = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (2).csv")
print(dataset.head(3))

print("--------------------------------------------------------------------------------------------------")
# Getting data of column "Married"
print(dataset['Married'].unique())
print(dataset['Married'].nunique())

# Filling nan values in 'Married' column
dataset['Married'].fillna(dataset["Married"].mode()[0], inplace = True)
print(dataset)

print("--------------------------------------------------------------------------------------------------")
print(dataset['Married'].isnull().sum())  # Checking for null values after filling

print("--------------------------------------------------------------------------------------------------")
# Getting data of column "Married" after fill null or nan values
print(dataset['Married'].unique())

print("--------------------------------------------------------------------------------------------------")
# Making Encoding order:-
en_data_ord = [['Yes', 'No']]
from sklearn.preprocessing import OrdinalEncoder
oen = OrdinalEncoder(categories=en_data_ord)
dataset['Married'] = oen.fit_transform(dataset[['Married']])
print(dataset)      # Checking dataset after encoding in 'Married' column











