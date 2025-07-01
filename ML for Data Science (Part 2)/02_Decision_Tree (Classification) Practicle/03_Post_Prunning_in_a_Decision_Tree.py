import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from mlxtend.plotting import plot_decision_regions
df = pd.read_csv("C:/Users/Yash/Downloads/social network ads.csv")
print(df.head())

print("-------------------------------------------------------------------------------------------------------------------------------")
print(df.isnull().sum())

print("-------------------------------------------------------------------------------------------------------------------------------")
'''Splitting the dataset'''
x = df.iloc[:,:-1]
y = df['Purchased']

'''Checking the nature of graph'''
sns.scatterplot(x = 'Age', y = 'EstimatedSalary', data = df, hue = 'Purchased')
plt.show()   # As you can see that your data is non-Linearly distributed

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x)
print(sc.transform(x))

'''Converting these transformed data into dataframe'''
x = pd.DataFrame(sc.transform(x), columns=x.columns)
print(x)

'''Training and Testing of Data'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42)
print(x_train, x_test)
print(y_train, y_test)

'''Making Decision Tree Model'''

'''Applying [Post-Prunning] to check which model depth(i) is best'''
from sklearn.tree import DecisionTreeClassifier
for i in range(1, 20):
    dtc = DecisionTreeClassifier(max_depth = i)
    dtc.fit(x_train, y_train)
    print(dtc.score(x_train, y_train), dtc.score(x_test, y_test), i)   # In output you can see 2 or 3 are the best models so use anyone of them

dtc = DecisionTreeClassifier(max_depth = 2)
dtc.fit(x_train, y_train)

print("---------------------------------------------------------------------------------------------------------------------------------------------------")


print(dtc.score(x_test, y_test)*100)
print(dtc.score(x_train, y_train)*100)

print(dtc.predict([[35, 20000]]))

'''Graphical analysis of this Decision Tree'''
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 20))
plot_tree(dtc)
plt.savefig("Demo.jpg")
plt.show()

print("-------------------------------------------------------------------------------------------------------------------------------")

'''Making Graph of plot decision Regions'''
plot_decision_regions(x.to_numpy(), y.to_numpy(), clf = dtc)
plt.show()


