import pandas as pd
df = pd.read_csv("C:/Users/Yash/OneDrive/Desktop/StudentPerformanceFactors.csv")
print(df.head())

print("--------------------------------------------------------------------------------------------------------------")
print(df.info())

print("--------------------------------------------------------------------------------------------------------------")
print(df.isnull().sum())

print("--------------------------------------------------------------------------------------------------------------")
'''As we can see the sleep_hours is object dtype because it has '6+' data in it that is considered as string or object'''
print(df['Sleep_Hours'].value_counts())   # Here you can see '6+' is available

print("--------------------------------------------------------------------------------------------------------------")
'''Replacing '6+' with '6' in the column 'Sleep_hours'''
df['Sleep_Hours'].replace('6+', "6", inplace = True)
print(df)

print("-------------------------------------------------------------------------------------------------------------")
'''Now check for '6+' of column 'Sleep_Hours' once again'''


print(df["Sleep_Hours"].value_counts())  # Here you can see 6+ has been replaced by 6

print("-------------------------------------------------------------------------------------------------------------")
print(df.info())

'''column 'sleep_hours' is still showing as object dtype'''
'''Now converting its dtype into (int dtype)'''
print("-------------------------------------------------------------------------------------------------------------")
df['Sleep_Hours'] = df['Sleep_Hours'].astype("int64")
print(df['Sleep_Hours'].dtype)    # Here you can see the datatype of column 'Sleep_Hours' is converted to int data_type



