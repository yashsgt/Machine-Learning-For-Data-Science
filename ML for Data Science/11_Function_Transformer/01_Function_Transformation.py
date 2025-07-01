'''Function Transformation:-> Function transformation in machine learning model refers to the process of applying mathematical
   or custom transformations to features in a dataset to improve model performance or preprocessing. Using FunctionTransformer
   in scikit-learn, you can define and apply these transformations seamlessly within a pipeline.
'''

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (2).csv") 
print(df)

print("-----------------------------------------------------------------------------------------------------")
print(df.isnull().sum())

print("-----------------------------------------------------------------------------------------------------")
df['Tests'].fillna(df['Tests'].mean(), inplace = True)
print(df)

print("-----------------------------------------------------------------------------------------------------")
print(df["Tests"].isnull().sum())

print("-----------------------------------------------------------------------------------------------------")
sns.distplot(df['Tests'])
plt.title('We can see it is non-normal distribution graph')             
plt.show()

print("-----------------------------------------------------------------------------------------------------")
'''First of all we will remove the outliers from column "tests" and then apply function transformer to convert this data 
   into a normal distribution'''
   
# Using IQR method to remove the outliers
q1 = df['Tests'].quantile(0.25)
q3 = df['Tests'].quantile(0.75)
IQR = q3 - q1
print("IQR is =", IQR)

min_range = q1 - (1.5*IQR)
max_range = q3 + (1.5*IQR)
print("The minimum range is =", min_range)
print("The maximum range is =", max_range)

'''Since, min_range is -ve and max_range is +ve so we will use max_range for the removal of outliers'''
df = df[df['Tests']<=max_range]
print(df)
sns.distplot(df['Tests'])
plt.title("We can see this is Distribution of the graph after removal of the outliers")
plt.show()

print("-----------------------------------------------------------------------------------------------------")
'''Converting this distribution into normal distribution by using function transformer'''
from sklearn.preprocessing import FunctionTransformer
ft = FunctionTransformer(func = np.log1p)
ft.fit(df[['Tests']])
print("Transformed data =", ft.transform(df[['Tests']]))

print("-----------------------------------------------------------------------------------------------------")
'''Making a new column 'Tests_tf' of the tranformed data'''
df['Tests_tf'] = ft.transform(df[['Tests']])
print(df)

print("-----------------------------------------------------------------------------------------------------")
'''Now compare both graph simultaneously for columns 'Tests' and 'Tests_tf' '''
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.distplot(df['Tests'])
plt.title('Before Function Tranformation')

plt.subplot(1, 2, 2)
sns.distplot(df['Tests_tf'])
plt.title('After Function Tranformation')

plt.show()  # Here you can see that the graph after function transformation is more towards normal distribution because
            # it has shorter tail due to removal of outlier than graph before Function tranformation
            
            
            


