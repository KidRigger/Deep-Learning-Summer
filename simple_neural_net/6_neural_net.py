# neural net with 2 inputs 2 nodes in hidden layer and one output (XOR)

import numpy as np

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def deriv(z):
    return(z*(1-z))

alpha = 0.1
np.random.seed(1234)
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0,1,1,0]]).T
w1 = np.random.random((2,2))
w2 = np.random.random((2,1))

for i in range(50000):
    z1 = np.dot(x,w1)
    o1 = sigmoid(z1)
    z2 = np.dot(o1,w2)
    o2 = sigmoid(z2)

    err = (y-o2)
    tmp2 = err*deriv(o2)
    inc2 = np.dot(o1.T,tmp2) * alpha
    w2+=inc2

    tmp1 = deriv(o1)*np.dot(tmp2,w2.T)
    inc1 = np.dot(x.T,tmp1) * alpha
    w1+=inc1

for i in range(len(x)):
    print('input : ', x[i], ' prediction : ', o2[i].round(3))
