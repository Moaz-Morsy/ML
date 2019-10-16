# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:26:56 2019

@author: ckrokad
"""
#SIMPLE LINEAR REGRATION 

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#spliting data into train and test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,test_size = 1/3,random_state = 0)

'''#features scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#fiting SLR to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train,sample_weight=0.56)

#prediction
Y_pred = regressor.predict(X_test) 

#visualising training set
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#visualising test set
plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()