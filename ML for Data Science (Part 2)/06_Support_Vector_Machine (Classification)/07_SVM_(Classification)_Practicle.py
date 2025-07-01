'''Types of SVM:->  (1.) Linear SVM: It follows y = mx + c (straight line) to separate the data
                    (2.) Non-Linear SVM: It converts the data into higher dimension after that y = mx + c is applied to separate the data
                    
'''

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv("C:/Users/Yash/Downloads/Placement (1) (1).csv")

print(df.head(3))

print(df.isnull().sum())

print(df.describe())

print(df.info())

print(df.drop(columns=['Student_ID'], inplace=True))

print("---------------------------------------------------------------------------------------------------------------------------------------------------")
print(df)

plt.figure(figsize=(5, 3))
sns.scatterplot(x = 'CGPA', y = 'IQ', data=df, hue = 'Placement')
plt.show()   # You can see it is linearly separable graph

x = df.iloc[:,:-1]
y = df['Placement']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x)
print(sc.transform(x))

'''Converting these transformed data into dataframe'''
x = pd.DataFrame(sc.transform(x), columns=x.columns)
print(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)

print(svc.score(x_test, y_test)*100)
print(svc.score(x_train, y_train)*100)
'''As there is a huge difference between the training and testing data'''
print("-------------------------------------------------------------------------------------------------------------------------------------------------------")

'''Applying [Post-Prunning] to check which model depth(i) is best'''
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=2)
dtc.fit(x_train, y_train)

# for i in range(1, 20):
#     dtc = DecisionTreeClassifier(max_depth = i)
#     dtc.fit(x_train, y_train)
#     print(dtc.score(x_train, y_train), dtc.score(x_test, y_test), i) 

print(dtc.score(x_test, y_test)*100)
print(dtc.score(x_train, y_train)*100)


plot_decision_regions(x.to_numpy(), y.to_numpy(), clf = dtc)
plt.show()