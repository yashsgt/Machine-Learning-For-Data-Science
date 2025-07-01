'''Removing Ioutliers using IQR method'''



import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (2).csv")
print(dataset)
print("---------------------------------------------------------------------------------------------------------")
print(dataset.shape)

print("---------------------------------------------------------------------------------------------------------")
# plt.figure(figsize=(10, 5))
sns.boxplot(x = "Recovered", data = dataset)
plt.show()

''' min_value or min_range = Q1 - (1.5*IQR)        below min_value there will be outlier         [IQR = Q3 - Q1]
    max_value or max_range = Q3 + (1.5*IQR)        above max_value there will be outlier
'''

'''Calculating Quantiles'''
# calculating first quantile (q1)
q1 = dataset['Recovered'].quantile(0.25)   
print("q1 =", q1)

# calculating 3rd quantile (q3)
q3 = dataset['Recovered'].quantile(0.75)
print("q3 =",q3)

IQR = q3 - q1
print("IQR =", IQR)

min_range = q1 - (1.5*IQR)
print("The minimum range is =", min_range)

max_range = q3 + (1.5*IQR)
print("The maximum range is =", max_range)

print("---------------------------------------------------------------------------------------------------------")
print(dataset.describe())

print("-------------------------------Removing Outliers from our dataset columns [Cases]----------------------------------------")

# Filter the dataset permanently based on the condition
dataset = dataset[dataset["Cases"] <= max_range]     #'''Taking all value below max range and ignoring outliers'''
# Now your dataset contains only the filtered rows
print(dataset)


print("---------------------------------------------------------------------------------------------------------")
print(dataset.shape)

sns.boxplot(x = "Recovered", data = dataset)  # plotting graph of column 'Cases' after removal of outliers
plt.show()


