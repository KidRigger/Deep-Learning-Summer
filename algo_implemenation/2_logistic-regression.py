import pandas as pd
import numpy as np
from sklearn import preprocessing,datasets,linear_model

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target



X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
X_train.shape
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
