import numpy as np
from NetworkNode import *

class NeuralNetwork:


    DEFAULT_LAYER = np.array([2, 1])
    def __init__(self, X, y, layers=DEFAULT_LAYER, loss_func=Functions.init_cross_entropy()):
        self.m_loss_func = loss_func
        self.m_X = X
        self.m_y = y

        self.m_layers = []
        number_of_weights_vector = layers[:-1]
        number_of_features = self.m_X.shape[1]
        number_of_weights_vector = np.concatenate((np.array([number_of_features]), number_of_weights_vector))

        vNodes = np.vectorize(NetworkNode.generic_init)
        print(number_of_weights_vector)

        for i in range(len(layers)):


            init_arr = vNodes(np.ones(layers[i], dtype=int) * number_of_weights_vector[i], )
            print(init_arr)
            self.m_layers.append(init_arr)

            print(self.m_layers)



    def fit(self, X_train, y_train, X_test, y_test):
        self.m_layers.reverse()
        for i in range(1000):
            loss_der = self.m_loss_func.der(X_train,y_train)

            for node in self.m_layers[0]:
                node.gradient_step()

            for layer in self.m_layers:
                pass
                #if i < 3 or i % 100 == 0:



    def __repr__(self):
        layers_string = ""
        for layer in self.m_layers:
            layers_string += str(len(layer)) + ', '
        return f"NeuralNetwork({len(self.m_layers)} layers= [{layers_string}])"

