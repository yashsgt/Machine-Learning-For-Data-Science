import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("C:/Users/Yash/Downloads/Iris.csv")
print(df.head())

print(df.drop(columns=["Id", "Species"], inplace=True))
print(df.head())

sns.pairplot(data=df)
plt.show()


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
plt.show()  # No. of Cluster will be 3 or 4 according to wcss process, it will be 3 according to Elbow value of the graph


kmn = KMeans(n_clusters=3)
df['Predict'] = kmn.fit_predict(df)    # making a column name 'predict'

print(kmn.labels_)
print(df.head())

'''Now we will find the exact value of no. of Clusters by Silhouette Score'''
from sklearn.metrics import silhouette_score
ss = silhouette_score(df, labels=kmn.labels_)
print(ss)

'''Checking the best Clusters'''

ss = []
no_of_clusters = [j for j in range(2, 21)]
for i in range(2, 21):
    km1 = KMeans(n_clusters=i)
    km1.fit(df)
    ss.append(silhouette_score(df, labels=km1.labels_))
    
plt.plot(no_of_clusters, ss)
plt.ylabel("Silhouette_score")
plt.xlabel("No. of clusters")
plt.xticks(no_of_clusters)   # Best silhouette score is at 3 so , no. of clusters = 3
plt.grid(axis='x')
plt.show()