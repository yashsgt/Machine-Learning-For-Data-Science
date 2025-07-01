'''An outlier is a value in a dataset that is noticeably different from most of the other values. It either stands out
   because it's much higher or much lower than the rest. Outliers can occur due to errors in data collection, unusual events,
   or natural variations. For example, in the numbers 3, 4, 5, 100, the value 100 is an outlier because it is much larger than
   the rest.
'''     




import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (2).csv")
print(dataset.head(3))

print("--------------------------------------------------------------------------------------------------------")
print(dataset.info())

print("--------------------------------------------------------------------------------------------------------")
print(dataset.describe())


print("--------------------------------------------------------------------------------------------------------")
print(dataset.isnull().sum())


print("--------------------------------------------------------------------------------------------------------")
dataset['Cases'].fillna(dataset['Cases'].mode()[0], inplace=True)
print(dataset)

print("--------------------------------------------------------------------------------------------------------")
print(dataset["Cases"].isnull().sum())   # Checking the null value after filling


print("--------------------------------------------------------------------------------------------------------")
dataset['Recovered'].fillna(dataset['Recovered'].mode()[0], inplace = True)
print(dataset)

print("--------------------------------------------------------------------------------------------------------")
print(dataset["Recovered"].isnull().sum())  # Checking the null values after filling

print("--------------------------------------------------------------------------------------------------------")
'''Making Box-plot for columns = 'Cases' and 'Recovered' to determine outliers'''
sns.boxplot(x = 'Cases', data = dataset)
plt.show()
sns.boxplot(x = 'Recovered', data = dataset)
plt.show()

sns.distplot(dataset['Cases'])
plt.show()