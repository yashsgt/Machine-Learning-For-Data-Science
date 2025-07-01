import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (1).csv")
print(df)

print("--------------------------------------------------------------------------------------------------------------")
print(df.isnull().sum())

print("--------------------------------------------------------------------------------------------------------------")
print(df.info())

print("--------------------------------------------------------------------------------------------------------------")
print(df.select_dtypes(include='float64'))

print("--------------------------------------------------------------------------------------------------------------")
# Ensure 'df' remains a DataFrame, and get columns with float64 data type
float_columns = df.select_dtypes(include='float64').columns
print("Columns name in list form:", float_columns)  # Returns the name of the columns with float datatype

print("--------------------------------------------------------------------------------------------------------------")
# Impute or filling the missing values in selected float columns
si = SimpleImputer(strategy='mean')
df[float_columns] = si.fit_transform(df[float_columns])
print(df[float_columns])  # Output the imputed columns and all missing values are filled


print("--------------------------------------------------------------------------------------------------------------")
new_df = pd.DataFrame(df[float_columns], columns = df.select_dtypes(include='float64').columns)
print(new_df)     # Output the imputed columns and all missing values are filled by the mean value of their corresponding columns


print("--------------------------------------------------------------------------------------------------------------")
print(new_df.isnull().sum())   # Here you can see there is no null values in new_df



print("--------------------------------------------------------------------------------------------------------------")
print(df['Deaths'].mean())   # Verifying the mean value of column Deaths















