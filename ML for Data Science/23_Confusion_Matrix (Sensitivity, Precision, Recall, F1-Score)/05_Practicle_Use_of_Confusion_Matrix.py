import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/placement (3).csv")
print(df.head(3))

print("-------------------------------------------------------------------------------------------------------------------------------------------")
x = df.iloc[:,:-1]
y = df['placed']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

print(lr.score(x_test, y_test)*100)

print("------------------------------------------------------Making Confusion Matrix-------------------------------------------------------------")

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
print(confusion_matrix(y_test, lr.predict(x_test)))    # Confusion matrix requires two parameters y_true, and y_pred and then we get Confusion matrix

print("-------------------------------------------Applying Graphical method on Confusion Matrix--------------------------------------------------")
cf = confusion_matrix(y_test, lr.predict(x_test))
sns.heatmap(cf, annot=True)
plt.show()
print("-------------------------------------------------------------------------------------------------------------------------------------------")
'''Checking Precision Score'''
print(precision_score(y_test, lr.predict(x_test))*100)
print("-------------------------------------------------------------------------------------------------------------------------------------------")
'''Checking Recall score'''
print(recall_score(y_test, lr.predict(x_test))*100)
print("-------------------------------------------------------------------------------------------------------------------------------------------")
'''Checking F1-Score'''
print(f1_score(y_test, lr.predict(x_test))*100)

