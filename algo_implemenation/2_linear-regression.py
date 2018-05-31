import pandas as pd
import numpy as np
from sklearn import preprocessing,datasets,linear_model

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# import some data 
iris = datasets.load_iris()
X = iris.data[:, :2]  
Y = iris.target



X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
X_train.shape
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, y_train)
linear.score(X_train, y_train)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(X_test)
print(predicted)
print(y_test)