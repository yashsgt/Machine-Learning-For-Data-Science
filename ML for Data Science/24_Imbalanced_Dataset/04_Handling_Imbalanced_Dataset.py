import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Yash/Downloads/social network ads.csv")
print(df.head(3))

print("------------------------------------------------------------------------------------------------------------------------------------------------")
print(df['Purchased'].value_counts())      # Checking the total no. of unique values for column = 'Purchased'

print("------------------------------------------------------------------------------------------------------------------------------------------------")
x = df.iloc[:,:-1]
y = df["Purchased"]

print("------------------------------------------------Training and Testing of Data---------------------------------------------------------")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

print("-------------------------------------------------Model of data-----------------------------------------------------------")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)*100)

print(lr.predict([[19, 19000]]))

print(lr.predict([[32,150000]]))

print(lr.predict([[27, 137000]]))   # It is 1 but giving zero because this data is biased by the majority data in dataset