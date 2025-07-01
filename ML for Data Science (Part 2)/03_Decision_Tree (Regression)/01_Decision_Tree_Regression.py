import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
df = pd.read_csv("C:/Users/Yash/Downloads/multiple_linear_regression_dataset (1).csv")
print(df.head())

print("------------------------------------------------------------------------------------------------------------------------------------------")

print(df.isnull().sum())


x = df.iloc[:,:-1]
y = df['income']

print("------------------------------------------------------------------------------------------------------------------------------------------")
sns.scatterplot(x = 'age', y = 'experience', data = df, hue = 'income')
plt.show()

sns.pairplot(data=df)
plt.show()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x)
print(sc.transform(x))

'''Converting these transformed data into dataframe'''
x = pd.DataFrame(sc.transform(x), columns=x.columns)
print(x)



print("------------------------------------------------------------------------------------------------------------------------------------------")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor, plot_tree
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)

print(dtr.score(x_test, y_test)*100)

print(dtr.score(x_train, y_train)*100)
plot_tree(dtr)
plt.show()
