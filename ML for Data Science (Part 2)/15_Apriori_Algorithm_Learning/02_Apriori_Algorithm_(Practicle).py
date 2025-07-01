import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

pd.set_option("display.max_rows", 200)

df = pd.read_csv("C:/Users/Yash/Downloads/groceries - groceries (2).csv")
print(df.head(3))

print(df.drop(columns=['Item 12', 'Item 13','Item 14','Item 15','Item 16','Item 17','Item 18','Item 19','Item 20','Item 21','Item 22','Item 23','Item 24','Item 25','Item 26','Item 27','Item 28','Item 29','Item 30','Item 31','Item 32'], inplace=True))
print(df.head())

print()
print(df.columns)

print("----------------------------------------------------------------------------------------------------------------------------------------------")

print(df['Item 1'][0])
print(type(df['Item 1'][0]))

print("----------------------------------------------------------------------------------------------------------------------------------------------")

print(df['Item 5'][0])
print(type(df['Item 5'][0]))

print("----------------------------------------------------------------------------------------------------------------------------------------------")
print(df.shape)

print("----------------------------------------------------------------------------------------------------------------------------------------------")
'''Printing Last row'''
print(df['Item 1'][9834])


'''Removing NAN values and making list of bought products by the customers'''
market = []
for i in range(0, df.shape[0]):
    customer = []
    for j in df.columns:
        if type(df[j][i]) == str:
            customer.append(df[j][i])
    market.append(customer)
    
print(market)  # So we have removed all nan values, i.e; In Market list we have the products only , there is no null values
               # In market list, we have a huge no of list of products
'''Now we will find the most frequently bought prodcut'''
import collections
a = collections.Counter(['a', 'a', 'a', 's', 's', 'd'])   # Counter function counts the number of data and makes a dictionary containing each type of products and its number
print(a)

print("\n")
print("======================================================================================================================================================")
print("\n")

'''Bringing all the list in a single list'''
l = []
for i in market:
    for j in i:
        l.append(j)   
print(l)

print("\n")
print("======================================================================================================================================================")
print("\n")

p = collections.Counter(l)
print(p)   # We get the dictionary containg each products and its number

print("\n")
print("======================================================================================================================================================")
print("\n")

'''Making a Dataframe containing each products'''
d = {'Item Name': p.keys(), "Values":p.values()}
df = pd.DataFrame(d)
print(df)

print("\n")
print("======================================================================================================================================================")
print("\n")

print(df.sort_values(by = ['Values'], ascending=False))

print("\n")
print("======================================================================================================================================================")
print("\n")

'''Applying Apriori algorithm'''
from mlxtend.preprocessing.transactionencoder import TransactionEncoder
tr = TransactionEncoder()
tr.fit(market)
print(tr.transform(market))

a = pd.DataFrame(tr.transform(market), columns= tr.columns_)
print(a)

print("\n")
print("======================================================================================================================================================")
print("\n")

from mlxtend.frequent_patterns import apriori
ap = apriori(a, min_support=0.05, max_len=3, use_colnames=True).sort_values(by=['support'], ascending=False)    #  min_support=0.05 means 5% support
print(ap)

# By changing hyperparameters like 'min_support' of apriori we found that '(whole milk, other vegetables)' is repeating maximum times as a most bought product




