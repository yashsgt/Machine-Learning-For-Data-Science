import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/Yash/Downloads/social network ads.csv")
print(df.head(3))

print("------------------------------------------------------------------------------------------------------------------------------------------------")
print(df['Purchased'].value_counts())      # Checking the total no. of unique values for column = 'Purchased'

x = df.iloc[:,:-1]
y = df['Purchased']

print("-----------------------------------------------------Balancing the Dataset using imblearn module(Under Sampling)-----------------------------------------")
from imblearn.under_sampling import RandomUnderSampler
ru = RandomUnderSampler()
ru_x, ru_y = ru.fit_resample(x, y)

print(ru_x)
print(ru_y)

print(ru_y.value_counts())   # Checking whether undersampling method working or not
                             # here you can see 0 is 143 and 1 is also 143 i.e; data is balanced according to Random Undersampling
                             
print("---------------------------------------------------Training and Testing of Under Sampling data-----------------------------------------")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ru_x, ru_y, test_size=0.20, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)*100)

print(lr.predict([[19, 19000]]))

print(lr.predict([[32,150000]]))

print(lr.predict([[27, 137000]])) 

print(lr.predict([[47,49000]]))

print(lr.predict([[48,41000]]))      # here you can see it is giving correct answer i.e; corresponding value = 1 is in dataset and it gives also the same in output
                                    # Here Our model is not biased beacause we have converted the data into a balanced dataset