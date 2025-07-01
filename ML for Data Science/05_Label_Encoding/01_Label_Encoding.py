'''Label encoding is a technique used to convert categorical data into numerical data by assigning a unique integer
   to each category. It is commonly used in machine learning preprocessing to make categorical data compatible with
   algorithms that require numerical inputs.'''


import pandas as pd
df = pd.DataFrame({'name' : ["Humans", "Tiger", 'Cat', "Lion", "Cow"]})
print(df)

print("----------------------------------------------------------------------------------------------------------------")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
print(le.fit_transform(df['name']))  # it will give our data into encoded form

print("----------------------------------------------------------------------------------------------------------------")
# Fiiting a column of enoded data 'en_name' with name to the dataset table
df['en_name'] = le.fit_transform(df['name'])
print(df)

print("----------------------------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------------------")
dataset = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (2).csv")
print(dataset)


print("----------------------------------------------------------------------------------------------------------------")
la = LabelEncoder()
print(la.fit(dataset['continent']))
print(la.transform(dataset['continent']))    # it is providing a unique integer value to each continent in the dataset while transforming


print("----------------------------------------------------------------------------------------------------------------")
'''Putting the encoded data 'continent' into the dataset table''' 
dataset['continent'] = la.transform(dataset["continent"])
print(dataset)




