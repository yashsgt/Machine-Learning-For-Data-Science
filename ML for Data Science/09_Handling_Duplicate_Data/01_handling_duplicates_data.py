'''[Duplicate Data]:-> Duplicate data in Python typically refers to rows or entries in a dataset that are repeated, meaning
   they have identical values across all columns. These duplicates can occur unintentionally during data collection
   or processing.'''

import pandas as pd
data = {"name":["a", "b", "c", "d", "a", "c"], "english":[8, 7, 5, 8, 8, 4], "hindi":[2, 3, 4, 5, 2, 6]}
df = pd.DataFrame(data)
print(df)
# In output you can see we have duplicate data of 1st row present in 5th row     

print("--------------------------------------------------------------------------------------------")
'''Check for duplicate data'''
print(df.duplicated())

print("--------------------------------------------------------------------------------------------")
# '''Fitting a column for duplicate data that predicts about duplicacy of data'''
# df['duplicated_data'] = df.duplicated()

print("--------------------------------------------------------------------------------------------")
print(df)

print("--------------------------------------------------------------------------------------------")
'''Removing of duplicates data'''
df.drop_duplicates(inplace=True)
print(df)   # Here you can see the duplicate data is removed

'''[Importance of removal of duplicated data]:-> Removing duplicate data ensures your analysis is accurate and unbiased,
   avoiding errors caused by repeated information. It also keeps your dataset efficient, clean, and easier to work with.'''
   
