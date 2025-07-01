'''DBSCAN Clustering Algorithm is used for the non-linearly separable dataset

   DBSCAN Clustering Algorithm is also useful in the detection of Outliers'''
   
   
   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=250, noise=0.05)
print(x)
print("--------------------------------------------------------------------------------------------------------------------------------------------------------")
print(y)

'''Loading dataset of 0th and 1st column of x-axis'''
df = {"data_1":x[:,0], "data_2":x[:,1]}

'''Converting this data df into dataframe'''
dataset = pd.DataFrame(df)
print(dataset)

'''Drawing graph between data_1 and data_2'''
sns.scatterplot(x = "data_1", y = "data_2", data=dataset)
plt.show()  # We got non-linear dataset

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.3, min_samples=5)
dataset['Predict'] = db.fit_predict(dataset)   # Making 'predict' column to know about the output of prediction

sns.scatterplot(x = "data_1", y = "data_2", data=dataset, hue='Predict')
plt.show() 