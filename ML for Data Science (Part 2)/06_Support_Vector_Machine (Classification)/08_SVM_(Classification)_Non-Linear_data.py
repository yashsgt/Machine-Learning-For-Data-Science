import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv("C:/Users/Yash/Downloads/placement (1).csv")
print(df)

plt.figure(figsize=(5, 3))
sns.scatterplot(x = 'cgpa', y = 'placement_exam_marks', data=df, hue='placed')
plt.show()
# You can see that it is a non-linearly separable dataset

x = df.iloc[:,:-1]
y = df['placed']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(x)
print(ss.transform(x))
x = pd.DataFrame(ss.transform(x), columns=x.columns)
print(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=10)

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(x_train, y_train)

print(svc.score(x_test, y_test)*100)
print(svc.score(x_train, y_train)*100)


plot_decision_regions(x.to_numpy(), y.to_numpy(), clf = svc)
plt.show()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

'''Applying Post Prunning to checking for the best fit model'''
for i in range(1, 21):
    dtc = DecisionTreeClassifier(max_depth=i)
    dtc.fit(x_train, y_train)
    print(i, dtc.score(x_train, y_train)*100, dtc.score(x_test, y_test)*100)

    