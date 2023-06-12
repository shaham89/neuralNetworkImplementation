import numpy as np
from NetworkNode import *

class NeuralNetwork:

    DEFAULT_LAYER = np.array([2, 1])
    def __init__(self, X, y, layers=DEFAULT_LAYER):
        self.X = X
        self.y = y

        self.m_layers = []
        number_of_weights_vector = layers[:-1]
        number_of_features = X.shape[1]
        number_of_weights_vector = np.concatenate((np.array([number_of_features]), number_of_weights_vector))

        vNodes = np.vectorize(NetworkNode.generic_init)
        print(number_of_weights_vector)

        for i in range(len(layers)):


            init_arr = vNodes(np.ones(layers[i], dtype=int) * number_of_weights_vector[i])
            print(init_arr)
            self.m_layers.append(init_arr)

            print(self.m_layers)

    def __repr__(self):
        layers_string = ""
        for layer in self.m_layers:
            layers_string += str(len(layer)) + ', '
        return f"NeuralNetwork({len(self.m_layers)} layers= [{layers_string}])"

