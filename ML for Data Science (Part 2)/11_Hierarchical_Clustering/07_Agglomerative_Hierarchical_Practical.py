'''[Agglomerative algorithm works on Linearly separable dataset]'''

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/Iris.csv")
print(df)

print(df.head(3))

'''Making it unlabeled dataset'''

print(df.drop(columns=['Id', 'Species'], inplace=True))
print(df.head(3))

'''Checking whether this dataset is linearly separable or not'''

sns.pairplot(data=df)
plt.show()

import scipy.cluster.hierarchy as sc
plt.figure(figsize=(20, 20))
sc.dendrogram(sc.linkage(df, method='single', metric='euclidean'))
plt.savefig('Dendogram.jpg')
plt.show()

# from graph it tells us to make two clusters

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, linkage='single')
df['Predict'] = ac.fit_predict(df)

print(df.head(3))

'''Checking CLuster'''
sns.pairplot(data=df, hue = 'Predict')
plt.show()

