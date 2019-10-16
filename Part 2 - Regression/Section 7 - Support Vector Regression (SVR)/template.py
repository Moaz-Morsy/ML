# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 13:01:09 2019

@author: ckrokad
"""

#template
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

#fitting the regressor model to the data set
#create regressor


#predict new result
new_val = np.array(6.5)
new_val = new_val.reshape(-1,1)
lin_reg.predict(new_val)

#visualising the regression result
plt.scatter(X, Y, color='red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff (regression model)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()
