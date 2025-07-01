import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/covid_19_NEW (1).csv")
print(df)

sns.heatmap(df.isnull())  # white color in graph show null values
plt.show()

'''You should never use data having null value more than or equal to 50% in a dataset
   You should use data having null values less than 50% in a dataset'''
   

