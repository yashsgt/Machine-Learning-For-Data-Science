''' z = (x - u)/sigma
    where x is the value, u is the mean of the data, and sigma is the standard deviation of the data.
    If z > 3 or z < -3 then it is an outlier.
    Useful data is in between range -3 < x < 3  or x = [-3, 3]
'''    
'''During first partition (u-sigma) to (u+sigma) We will have 68.26% of the data.
   During second partition (u-2*sigma) to (u+2*sigma) We will have 95.44% of the data.
   During third partition (u-3*sigma) to (u+3*sigma) We will have 99.73% of the data.
   So, if z > 3 or z < -3 then it is an outlier.
'''  

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (2).csv")
print(dataset)

print("---------------------------------------------------------------------------------------------------------")
print(dataset.isnull().sum())  # checking null values in the dataset

print("---------------------------------------------------------------------------------------------------------")
print(dataset.describe())  # checking the dataset description

print("---------------------------------------------------------------------------------------------------------")
sns.boxplot(x = "Recovered", data = dataset)  # plotting graph of column 'Cases' before removal of outliers
plt.show()


print("---------------------------------------------------------------------------------------------------------")
sns.distplot(dataset['Recovered'])  # plotting graph of column 'Recovered' before removal of outliers
plt.show()

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(dataset.shape)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

'''Removing the outliers using direct method'''
min_range = dataset['Recovered'].mean() - 3*dataset['Recovered'].std()   # min_range = [mean - 3rd std dev.]
max_range = dataset['Recovered'].mean() + 3*dataset['Recovered'].std()
print("The minimum range is =", min_range)
print("The maximum range is =", max_range)
print("-------------------------------------------------------------------------------------")
new_data = dataset[dataset['Recovered'] <= max_range]
print(new_data)  # this dataset doesn't contain any outlier
print("Shape of dataset using direct method: ",new_data.shape)     

'''Checking whether all outliers has been removed or not'''
sns.boxplot(x = 'Recovered', data = new_data)
plt.show()







print("__________________________________________________________________________________________________________________________________________________")
print("__________________________________________________________________________________________________________________________________________________")
print("__________________________________________________________________________________________________________________________________________________")
print("__________________________________________________________________________________________________________________________________________________")


'''Removing outliers using Z-Score method'''
z_score = (dataset['Recovered'] - dataset['Recovered'].mean())/dataset['Recovered'].std()    # z_score = (x - u)/std dev
print(z_score)

print("z_score greater than 3: ", z_score>3)
print("z_score less than 3: ", z_score<3)

# Placing the z-score as a column in our original dataset
dataset['z_score'] = z_score
print(dataset)

# Removng outliers
dataset = dataset[dataset['z_score']<3]  # considering dataset below 3rd z-score
print(dataset)
print("Shape of dataset using z_score method: ",dataset.shape)



