'''[Training]: This involves using a dataset (called the training set) to teach a machine learning model how to make
   predictions or decisions. The model learns patterns, relationship and rules from the data provided during training.

   [Testing]:  After training, the model is evaluated using a separate dataset (called the testing set) that it hasn't
   seen before. This helps in assessment how well the model performs in making prediction no new, unseen data.

'''


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Yash/Downloads/Boston.csv")
print(df)
print("--------------------------------------------------------------------------------------------------------------")

print(df.shape)

'''Note:-> During Supervised learning you have to train and test the data 
           During Unsupervised learning you don't have to train and test the data'''
           
'''Let's separate the input and output features of the data'''

input_data = df.iloc[:,:-1]  # here you can see that the last column is removed
print(input_data)

print("--------------------------------------------------------------------------------------------------------------")
output_data = df['medv']
print(output_data)

print("--------------------------------------------------------------------------------------------------------------")


'''Now do train and test'''
from sklearn.model_selection import train_test_split
# test_train_split gives you four outputs in which 2 for training and testing of input data and other 2 for training and testing of output data
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.25)     # x is for input data and y is for output data
# print(x_train, x_test, y_train, y_test)    

'''Checking the split of data into x and y'''
print("--------------------------------------------------------------------------------------------------------------")
print(x_train)
print(x_test)

print("--------------------------------------------------------------------------------------------------------------")
print(y_train)
print(y_test)

print("--------------------------------------------------------------------------------------------------------------")
'''By compairing the shape of original data to the splitted data into x and y. You can see that data is splitted'''
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

