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



print("----------------------------------------------------------------------------------------------------------------------------------------")
'''Making Decision Tree Model'''
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()  
dtc.fit(x_train, y_train)

print(dtc.score(x_test, y_test)*100)    # Checking the accuracy of testing data

print(dtc.score(x_train, y_train)*100)    # Checking the accuracy of training data
''' You can see there is huge difference between the accuracy of training and testing dataset so, this is the case of Overfitting of model
    so to avoid the risk of overfitting we use Prunning (Pre and Post Prunning)
'''
print("---------------------------------------------------------[Applying Pre-Prunning]----------------------------------------------------------")

dtc = DecisionTreeClassifier(max_depth = 5)   # This parameter 'max_depth' is used during pre-prunning
dtc.fit(x_train, y_train)

print(dtc.score(x_test, y_test)*100)    # Checking the accuracy of testing data

print(dtc.score(x_train, y_train)*100)    # Checking the accuracy of training data
''' You can see the difference between the accuracy of training and testing dataset is much more reduced by pre-prunning,
    also the risk of overfitting also avoided
'''
print("----------------------------------------------------------------------------------------------------------------------------------------")

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
