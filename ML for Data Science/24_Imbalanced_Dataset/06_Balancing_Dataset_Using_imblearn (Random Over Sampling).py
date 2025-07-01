import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Yash/Downloads/social network ads.csv")
print(df.head(3))

print("------------------------------------------------------------------------------------------------------------------------------------------------")
print(df['Purchased'].value_counts())      # Checking the total no. of unique values for column = 'Purchased'

x = df.iloc[:,:-1]
y = df['Purchased']

print("-----------------------------------------------------Balancing the Dataset using imblearn module(Over Sampling)-----------------------------------------")
from imblearn.over_sampling import RandomOverSampler
ro = RandomOverSampler()
ro_x, ro_y = ro.fit_resample(x, y)

print(ro_x)
print(ro_y)

print(ro_y.value_counts())   # Checking whether oversampling method working or not
                             # here you can see 0 is 257 and 1 is also 257 i.e; data is balanced according to Random Oversampling
                             
print("---------------------------------------------------Training and Testing of Over Sampling data-----------------------------------------")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ro_x, ro_y, test_size=0.20, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)*100)

print(lr.predict([[19, 19000]]))

print(lr.predict([[32,150000]]))

print(lr.predict([[27, 137000]])) 

print(lr.predict([[47,49000]]))

print(lr.predict([[48,41000]])) 