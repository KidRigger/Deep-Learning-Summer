import numpy as np
import pandas as pd

df=pd.read_csv('/home/nishchay/Documents/Arcon/Day6/winequality-red.csv')
df=pd.get_dummies(df,columns=['quality'])
df.drop(['citric acid','residual sugar','pH','free sulfur dioxide','quality_3','quality_8'], axis = 1, inplace = True)
X=df.iloc[:,0:7].values
y=df.iloc[:,7:11].values
X = X/np.amax(X, axis=0)
#xPredicted = xPredicted/np.amax(xPredicted, axis=0)
#y = y/100

class Neural_Network(object):
  def __init__(self):

    self.inputSize = 7
    self.outputSize = 4
    self.hiddenSize = 10

    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 

  def predict(self):
    print ('Predicting ...')
    for i in range(0,X.shape[0]):
        l1 = 1/(1 + np.exp(-(np.dot(X[i], NN.W1))))
        l2 = 1/(1 + np.exp(-(np.dot(l1, NN.W2))))
        a = np.round(l2,3)
        print('a: '+str(a))
        np.savetxt("foo.csv", a, delimiter=",")

NN = Neural_Network()

learning_rate = 0.2 # slowly update the network
for epoch in range(10000):
    for i in range(0,X.shape[0]):
        row=X[i][np.newaxis]
        l1 = 1/(1 + np.exp(-(np.dot(row, NN.W1))))# sigmoid function
        l2 = 1/(1 + np.exp(-(np.dot(l1, NN.W2))))
        er = (abs(y[i] - l2)).mean()
        l2_delta = (y[i][np.newaxis] - l2)*(l2 * (1-l2))
        l1_delta = l2_delta.dot(NN.W2.T) * (l1 * (1-l1))
        NN.W2 += l1.T.dot(l2_delta) * learning_rate
        NN.W1 += row.T.dot(l1_delta) * learning_rate
    #print ('Error:', er)
NN.predict()
