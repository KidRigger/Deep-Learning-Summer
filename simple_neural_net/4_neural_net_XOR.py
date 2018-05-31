import numpy as np

X = np.array(([0,1], [1,1],[0,0],[1,0]), dtype=float)
y = np.array(([1], [0],[0],[1]), dtype=float)
Pred = np.array(([1,0],[0,1],[1,1],[0,0]), dtype=float)

class Neural_Network(object):
  def __init__(self):
    self.i= 2
    self.o= 1
    self.h= 5
    self.W1 = np.random.randn(self.i, self.h)
    self.W2 = np.random.randn(self.h, self.o) 
  def forward(self, X):
    self.z = np.dot(X,self.W1) 
    self.z2 = self.sigmoid(self.z)
    self.z3 = np.dot(self.z2, self.W2)
    o = self.sigmoid(self.z3)
    return o
  def sigmoid(self, s):
    return 1/(1+np.exp(-s))
  def sigmoidPrime(self, s):
      return s * (1 - s)
  def backward(self, X, y, o):
    self.o_error = ((y-o))
    self.o_delta = self.o_error*self.sigmoidPrime(o) 
    self.z2_error = self.o_delta.dot(self.W2.T)
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) 
    self.W1 += X.T.dot(self.z2_delta)
    self.W2 += self.z2.T.dot(self.o_delta)
  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)
  def predict(self):
    print ("Input (scaled):" + str(Pred))
    print ("Output:" + str(self.forward(Pred)))
NN = Neural_Network()
for i in range(10000):
  NN.train(X, y)
NN.predict()
