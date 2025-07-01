import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
df = pd.read_csv("C:/Users/Yash/Downloads/placement (1).csv")
print(df.head(4))

sns.scatterplot(x = 'placement_exam_marks', y = 'cgpa', data=df, hue='placed')
plt.show()  # Here you can see that it is not a linearly separable dataset

x = df.iloc[:,:-1]
y = df['placed']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)*100)

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.to_numpy(), y.to_numpy(), clf = lr)   # converting x and y into numpy array for plotting this graph
plt.show()















