import pandas as pd
import numpy as np
from numpy import random,dot,exp

class Layer():
    def __init__(self, no_neurons, inputs_per_neuron):
        self.weights = 2 * random.random((inputs_per_neuron, no_neurons)) - 1


class NN():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __diff_sigmoid(self, x):
        return x * (1 - x)

    def train(self, training_ip, training_op, training_iterations):
        for iteration in range(training_iterations):
            output_from_layer_1, output_from_layer_2 = self.fwd(training_ip)

            layer2_error = training_op - output_from_layer_2
            layer2_delta = layer2_error * self.__diff_sigmoid(output_from_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.weights.T)
            layer1_delta = layer1_error * self.__diff_sigmoid(output_from_layer_1)

            layer1_adjustment = training_ip.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            self.layer1.weights += 0.1*layer1_adjustment
            self.layer2.weights += 0.1*layer2_adjustment

    def fwd(self, inputs):

        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.weights))
        print(output_from_layer2)
        return output_from_layer1, output_from_layer2
if __name__ == "__main__":

    np.random.seed(1)
    dataset = pd.read_csv('wine2.csv')
    values = list(dataset.columns.values)
    y = dataset[values[-3:]]

    y = np.array(y, dtype=np.float32)
    X = dataset[values[0:-3]]
    X = np.array(X, dtype=np.float32)

    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]

    test_size = 10
    X_test = X_values[-test_size:]
    X_train = X_values[:-test_size]
    y_test = y_values[-test_size:]
    y_train = y_values[:-test_size]

    layer1 = Layer(4, 13)
    layer2 = Layer(3,4)

    neural_network = NN(layer1, layer2)

    neural_network.train(X_train, y_train, 1000)
    y_op1,yop_2 = neural_network.fwd(X_test)
    for i in range(len(yop_2)):
        print(yop_2[i],"     ",y_test[i])
