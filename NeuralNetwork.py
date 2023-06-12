import numpy as np
from NetworkNode import *

class NeuralNetwork:

    DEFAULT_LAYER = np.array([2, 1])
    def __init__(self, X, y, layers=DEFAULT_LAYER):
        self.X = X
        self.y = y

        self.m_layers = np.array(len(layers), dtype=object)
        number_of_weights_vector = layers[:-1]
        number_of_features = X.shape[1]
        number_of_weights_vector = np.concatenate((np.array([number_of_features]), number_of_weights_vector))

        vNodes = np.vectorize(NetworkNode.generic_init)
        print(number_of_weights_vector)

        for i in range(len(layers)):

            init_arr = number_of_weights_vector[i]

            self.m_layers[i] = vNodes(init_arr)

            print(self.m_layers)

