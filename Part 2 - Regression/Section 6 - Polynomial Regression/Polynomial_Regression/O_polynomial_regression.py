# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 20:06:12 2019

@author: ckrokad
"""
#Polynomial regression

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#spliting data into train and test datasets
'''from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,test_size = 1/3,random_state = 0)'''

'''#features scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#fitting linear regression to the dataset
from sklearn.linear_model  import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

# Visualising the linear regression result
plt.scatter(X, Y, color='red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff (Linear regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

#visualising the polynomial regression result 
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title('Truth or Bluff (Polynomial regression)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

#predict new value by linear
new_val = np.array(6.5)
new_val = new_val.reshape(-1,1)
lin_reg.predict(new_val)

#predict new value by polynomial
lin_reg_2.predict(poly_reg.fit_transform(new_val))




















