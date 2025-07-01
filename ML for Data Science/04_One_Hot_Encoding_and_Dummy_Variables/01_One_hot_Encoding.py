'''Encoding is a process of conversion of data from one form to another form
   e.g:-> One-Hot-Encoding is used to convert the Categorical data into Numerical data'''
   
import pandas as pd
df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (2).csv")
print(df)

print("---------------------------------------------------------------------------------------------------------------")
print(df.head(5))

print("---------------------------------------------------------------------------------------------------------------")
'''Filling the missing values in Categorical data Such as Gender and Married'''

df['Married'].fillna(df['Married'].mode()[0], inplace=True)
print(df['Married'])

df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)
print(df['Gender'])


print("---------------------------------------------------------------------------------------------------------------")
print(df['Gender'].isnull().sum())   # Checking the null values for Gender after filling all null values

print("---------------------------------------------------------------------------------------------------------------")
print(df['Married'].isnull().sum())    # Checking the null values for Married after filling all null values

print("---------------------------------------------------------------------------------------------------------------")
'''One-Hot-Encoding'''
from sklearn.impute import SimpleImputer
en_data = df[['Gender', 'Married']]
print(pd.get_dummies(en_data))
'''The get_dummies function in Python, specifically in pandas, is used to convert categorical data into numerical data by
   creating dummy or indicator variables. This process is essential for machine learning models, as they typically require
   numerical inputs.'''

print("---------------------------------------------------------------------------------------------------------------")
print(pd.get_dummies(en_data).info())

print("---------------------------------------------------------------------------------------------------------------")
'''But we want our data in numerical form'''
from sklearn.preprocessing import OneHotEncoder  # sklearn is a module in which a file name preprocessing is there and it contain one-hot-encoder class

ohe = OneHotEncoder()

print("Sparse Matrix")                # only fit() can just train your data it can't transform your data
print(ohe.fit_transform(en_data))      # fit_transform() aapke data ko dekhta hai samajhta hai aur sklearn ki algorithm ko apply karke aapke data ko tranform kar deta hai(categorical data to Numerical data)
                                    # Sparse matrix give us the data in matrix-form filled with 1 and 0 which is the basis of one-hot-encoding work
print("---------------------------------------------------------------------------------------------------------------")
'''Converting this matrix data into array form'''
print(ohe.fit_transform(en_data).toarray())

print("---------------------------------------------------------------------------------------------------------------")
'''Converting the array form of data into DataFrame'''
arr = ohe.fit_transform(en_data).toarray()
new_data = pd.DataFrame(arr, columns=['Gender_Male', 'Gender_Female', 'Married_No', 'Married_Yes'])
print(new_data)   # Here you can see that encoding of your data is done

print("---------------------------------------------------------------------------------------------------------------")
'''If you want to drop the first column of new encoded dataset'''
ohe = OneHotEncoder(drop = 'first')
arr = ohe.fit_transform(en_data).toarray()
print(arr)  # Here you can see your first column(and other column of other category will also be remove according to its first most value of first column) 
            # Here Gender_Male and Married_No are removed
            
            
            
            

            
            
            

