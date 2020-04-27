# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 10:13:14 2019

@author: Kapil
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
#Read the data
#columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data=pd.read_csv("C:/Users/Kapil/Documents/Machine Learning/housing.csv")
#Dependent data
print("Dependent data")
prices=data['MEDV']
print(prices)

#Independent data
features=data.drop('MEDV',axis=1)
print(features)

#Success
print("Bostan housing dataset has {} data points with {} variables each.".format(*data.shape))

#Summary of Statistics
prices_description=prices.describe()
print(prices_description)
print("\n")

import matplotlib.pyplot as plt
import math

a=np.array(features['RM'])
x=a.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, prices, test_size = .2, random_state = 42)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""# Predicting the Test set results
y_pred = regressor.predict(X_test)"""

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs RM')
plt.xlabel('RM')
plt.ylabel('PRICE')
plt.show()

"""# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()"""

#LSTAT vs prices
a=np.array(features['LSTAT'])
x=a.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, prices, test_size = .2, random_state = 42)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs LSTAT')
plt.xlabel('LSTAT')
plt.ylabel('PRICE')
plt.show()

#PTRATIO vs prices
a=np.array(features['PTRATIO'])
x=a.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, prices, test_size = .2, random_state = 42)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs PTRATIO')
plt.xlabel('PTRATIO')
plt.ylabel('PRICE')
plt.show()

