'''[Standardization]: It is a very effective technique which re-scales a feature value so that it has distribution
   with 0 mean value and variance equals to 1
   
   Xnew = (Xi - Xmean)/standard deviation           where Xmean is the new data formed after standardization
''' 

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (2).csv")

print(df)
print("------------------------------------------------------------------------------------------------------------")

print(df.head(3))
print("------------------------------------------------------------------------------------------------------------")

print(df.isnull().sum())
print("------------------------------------------------------------------------------------------------------------")

df['population'].fillna(df["population"].mean(), inplace = True)
print(df)

print(df["population"].isnull().sum())  # checking for null value after filling

print("-----------------------------------------------------------------------------------------------------------")
'''plotting distplot for population column'''
sns.distplot(df["population"])
plt.show()

print("----------------------------------------------------------------------------------------------------------")
print(df.describe())


print("--------------------------------Scaling of the data by Standardization------------------------------------")
from sklearn.preprocessing import StandardScaler   
ss = StandardScaler()    # making an object ss as standard scaler
ss.fit(df[['population']])
print("Transformed data in array form: ", ss.transform(df[['population']]))

df['population'] = ss.transform(df[['population']])   # Replacing the population column with transformed data 
print(df) 

df['population_ss'] = pd.DataFrame(ss.transform(df[['population']]), columns=['x'])  # Making a new column ['population_ss'] of tranformed data
print(df)

print("---------------------------------------------------------------------------------------------------------")
print(df.describe())

print("---------------------------------------------------------------------------------------------------------")
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)   # 1 rows 2 columns and 1st position
plt.title("Before")
sns.distplot(df['population'])

plt.subplot(1, 2, 2)  # 1 rows 2 columns and 2nd position
plt.title("After")
sns.distplot(df["population_ss"])  # checking the nature of graph

plt.show()

'''By feature-scaling the magnitude of the data is reduced but it's nature remains the same and all these you can see on
   the graphs'''
   
   
   
