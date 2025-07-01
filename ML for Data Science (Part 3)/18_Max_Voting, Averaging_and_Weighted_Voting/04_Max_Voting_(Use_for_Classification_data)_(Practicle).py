import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=1000, noise=0.2)

data = {"x1":x[:,0], "x2":x[:,1], "y":y}
df = pd.DataFrame(data)
print(df)

sns.scatterplot(x = 'x1', y = 'x2', data=df, hue='y')
plt.show()


x_a = df.iloc[:,:-1]
y_a = df['y']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_a, y_a, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
print(dtc.score(x_train, y_train)*100, dtc.score(x_test, y_test)*100)

sv = SVC()
sv.fit(x_train, y_train)
print(sv.score(x_train, y_train)*100, sv.score(x_test, y_test)*100)

gnb = GaussianNB()
gnb.fit(x_train, y_train)
print(gnb.score(x_train, y_train)*100, gnb.score(x_test, y_test)*100)

'''Now making the best model by the combination of all used models'''

from sklearn.ensemble import VotingClassifier
li = [('dt1', DecisionTreeClassifier()), ('sv1', SVC()),('gnb', GaussianNB())]

vc = VotingClassifier(li, weights=[9, 8, 9])   # By changing weights we can improve the accuracy of our best model
vc.fit(x_train, y_train)

print(vc.score(x_train, y_train)*100, vc.score(x_test, y_test)*100)


'''Now Let's see how Voting Classifier(gives the best model) works by using the combination of models'''

prd = {'dtc':dtc.predict(x_test), 'sv':sv.predict(x_test), 'gnb':gnb.predict(x_test), 'vc':vc.predict(x_test)}

df2 = pd.DataFrame(prd)
print(df2)