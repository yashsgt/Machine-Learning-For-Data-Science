'''Normalization:-> It is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0
   and 1. It is also known as Min-Max scaling
   Xnew = (Xi - min(X))/(max(X) - min(X))
   
   e.g: Xi = 2, 3, 4, 5, 6, 7, 8, 9
   where min(X) is 2 and max(X) is 9'''
   
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (2).csv")
print(df.head(3))
print("-------------------------------------------------------------------------------------------------------")

print(df.isnull().sum())
print("-------------------------------------------------------------------------------------------------------")

df['Cases'].fillna(df["Cases"].mode()[0], inplace = True)
print(df)
print("-------------------------------------------------------------------------------------------------------")

print(df["Cases"].isnull().sum())  # checking for nul values after filling
print("-------------------------------------------------------------------------------------------------------")

print(df.describe())
print("-------------------------------------------------------------------------------------------------------")

sns.distplot(df['Cases'])
plt.show()

print("---------------------------Scaling of the data by Normalization(or MinMaxScalar)-----------------------")
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
ms.fit(df[["Cases"]])
print("Transform data in array form:- ", ms.transform(df[["Cases"]]))
df['Cases'] = ms.transform(df[['Cases']])  # Replacing the 'Cases' column with transformed data
print(df)

'''Making a new column ['Cases_ms'] of tranformed data'''
df["Cases_ms"] = pd.DataFrame(ms.transform(df[["Cases"]]))
print(df)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Before")
sns.distplot(df["Cases"])

plt.subplot(1, 2, 2)
plt.title("After")
sns.distplot(df["Cases_ms"])

plt.show()


