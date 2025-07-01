'''In [K-Mean clustering];  K-Mean++ is used to decide the clustering point much far away from each other so as to get the best number of clusters or groups

   or we can say that K-Mean++ helps in best clustering
   
   Also Elbow point in Graph (b/w WCSS and no. of clusters) give us the best no. of clusters'''

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/Iris.csv")
print(df)

print(df.head(3))
   
   
print(df.isnull().sum())

'''Removing Columns = [Id , Species] to make it unlabeled dataset (which don't have the output)'''

print(df.drop(columns=['Id', "Species"], inplace = True))

print(df.head()) # Now you can see that it is a labeled dataset

sns.heatmap(df.corr(), annot=True)
plt.show()

sns.pairplot(data=df)  # Checking whether our data is linearly separable or not becoz K_mean clustering is applied only when your dataset is linearly separable
plt.show()

'''In Unsupervised ML We don't use Train-Test-Split beacuse we don't have output data , We have unlabeled dataset'''

'''[Applying K-Mean Clustering]'''

from sklearn.cluster import KMeans
wcss = []
for i in range(2, 21):
   km = KMeans(n_clusters = i, init  = "k-means++")
   km.fit(df)
   wcss.append(km.inertia_)       # inertia_ is the value of wcss
   
   
'''Plotting graph b/w wcss and no. of clusters'''
plt.figure(figsize=(10, 5))
plt.plot([i for i in range(2, 21)], wcss, marker = 'o')
plt.xlabel("No. of Clusters")
plt.xticks([i for i in range(2, 21)])    # Improving x-axis curve
plt.ylabel("WCSS")
plt.grid(axis='x')
plt.show()   # from the graph you can see the elbow point is at point no. 3 so best no. of clusters = 3


kmn = KMeans(n_clusters=3)
df['Predict'] = kmn.fit_predict(df)    # making a column name 'predict'

print(df.head())

sns.pairplot(data=df, hue = "Predict")
plt.savefig('Predict.jpg')
plt.show()

print("========================================================================================================================================================")

'''Now checking how our original data behaves in clustering formation'''
org_data = pd.read_csv("C:/Users/Yash/Downloads/Iris.csv")
print(org_data)

print(org_data.drop(columns=['Id'], inplace = True))
print(org_data.head(3))

sns.pairplot(data=org_data, hue='Species')
plt.savefig('org_data.jpg')
plt.show()   