import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/social network ads.csv")
print(df.head())

print(df.isnull().sum())

sns.scatterplot(x = 'Age', y = 'EstimatedSalary', data = df, hue = 'Purchased')
plt.show()

x = df.iloc[:,:-1]
y = df['Purchased']



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x)
print(sc.transform(x))

x = pd.DataFrame(sc.transform(x), columns = x.columns)
print(x)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

print(knn.score(x_test, y_test)*100)
print(knn.score(x_train, y_train)*100)

print(knn.predict([[-1.781797, -1.490046]]))

'''For getting right value of neighbor to get best fit model'''
# for i in range(1, 30):
#     knn1 = KNeighborsClassifier(n_neighbors=i)
#     knn1.fit(x_train, y_train)
#     print(i, knn1.score(x_train, y_train)*100, knn1.score(x_test, y_test)*100)


'''For checking the decision region'''
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.to_numpy(), y.to_numpy(), clf = knn)
plt.show()