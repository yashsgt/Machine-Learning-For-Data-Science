import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv(r"C:/Users/Yash/Downloads/placement (3).csv")
print(df.head(4))
print("---------------------------------------------------------------------------------------------------------------------------------------------")
'''Checking data whether it is normal distribution or not (Checking for each columns or each parameters in a dataset)'''
sns.kdeplot(data = df['cgpa'])
plt.show()
# They are showing normal distribution approximately
sns.kdeplot(data = df['resume_score'])
plt.show()

sns.kdeplot(data = df["placed"])
plt.show()
print("---------------------------------------------------------------------------------------------------------------------------------------------")
print(df.isnull().sum())   # Checking for null values

print("---------------------------------------------------------------------------------------------------------------------------------------------")

plt.figure(figsize=(5,3))
sns.scatterplot(x = 'cgpa', y = 'resume_score', data=df, hue = 'placed')
plt.show()   # Here you can see that this data is linearly separable

print("---------------------------------------------------------------------------------------------------------------------------------------------")
x = df.iloc[:,:-1]
y = df['placed']

print("---------------------------------------------------------------------------------------------------------------------------------------------")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

print("---------------------------------------------------------------------------------------------------------------------------------------------")

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)*100)

print("----------------------------------------------------Applying Naive Bayes Algorithm-----------------------------------------------------------")

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
'''Here according to this dataset that shows normal distribution so GaussianNB algoritm will be applied, but we will check for each algoritm'''

'''[GaussianNB algoritm]'''
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# Checking the accuracy of GaussianNB algorithm for input(x) and output(x) data or training and testing data
print(gnb.score(x_train, y_train)*100)
print(gnb.score(x_test, y_test)*100)

print(gnb.predict([[8.14, 6.52]]))

print("---------------------------------------------------------------------------------------------------------------------------------------------")

'''[BernoulliNB algorithm]'''
bnb = BernoulliNB()
bnb.fit(x_train, y_train)

# Checking the accuracy of BernoulliNB algorithm for input(x) and output(x) data or training and testing data
print(bnb.score(x_train, y_train)*100)
print(bnb.score(x_test, y_test)*100)

print("---------------------------------------------------------------------------------------------------------------------------------------------")

'''MultinomialNB'''
mnb = MultinomialNB()
mnb.fit(x_train, y_train)

print(mnb.score(x_train, y_train)*100)
print(mnb.score(x_test, y_test)*100)


# Here from all other algorithm, we can see that GaussianNB algorithm gives highest accuracy of all because we have the normal distribution dataset

print("---------------------------------------------------------------------------------------------------------------------------------------------")
'''Checking for decision region for GaussianNB algorithm'''
plot_decision_regions(x.to_numpy(), y.to_numpy(), clf = gnb)
plt.show()    # Here you can see GaussianNB algorithm is working perfectly for this normal distribution dataset.