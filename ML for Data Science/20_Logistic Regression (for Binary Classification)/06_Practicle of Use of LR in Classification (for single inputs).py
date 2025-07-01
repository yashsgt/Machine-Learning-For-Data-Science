import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("C:/Users/Yash/Downloads/social network ads.csv")

print(df)
print("----------------------------------------------------------------------------------------------------------------------------------")
print(df.head(4))
print("----------------------------------------------------------------------------------------------------------------------------------")
df.drop(columns=['EstimatedSalary'], inplace = True)   # Dropping one column so that to make one input
print(df)

print("----------------------------------------------------------------------------------------------------------------------------------")
'''Checking our data whether it is suitable for Logistic Regression or not'''
plt.figure(figsize=(5, 3))
plt.xlabel("Age")
plt.ylabel("Purchased")
sns.scatterplot(x = "Age", y = "Purchased", data = df)
plt.show()    # Here you can see that your data is fit for LR because in graph answer in 1 or 0 (in Binary Terms)

print("----------------------------------------------------------------------------------------------------------------------------------")
'''Splitting of dataset into inputs(independent variables) and outputs(dependent variables)'''
x = df.iloc[:,:-1]
y = df['Purchased']

'''Training and testing of data'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)


from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression()
Lr.fit(x_train, y_train)


print("----------------------------------------------------------------------------------------------------------------------------------")
print(Lr.score(x_test, y_test)*100)


print("----------------------------------------------------------------------------------------------------------------------------------")
'''predicting data'''
print(Lr.predict([[46]]))   # It will predict whether age 46 has purchased or not


print("----------------------------------------------------------------------------------------------------------------------------------")
'''Drawing the prediction line'''

plt.figure(figsize = (5, 3))
plt.xlabel("Age")
plt.ylabel("Purchased")
sns.scatterplot(x = "Age", y = "Purchased", data = df)
sns.lineplot(x = "Age", y = Lr.predict(x), data = df, c = 'red')   # Drawing the prediction line
plt.show()  







