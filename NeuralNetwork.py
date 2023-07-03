import numpy as np

from Layer import Layer
from NetworkNode import *

class NeuralNetwork:


    DEFAULT_LAYER = np.array([2, 1])
    def __init__(self, X, y, layers=DEFAULT_LAYER, loss_func=Functions.init_cross_entropy()):
        self.m_loss_func = loss_func
        self.m_X = X
        self.m_y = y

        self.m_layers = []

        self.m_layers.append(Layer.input_layer_init(self.m_X))
        print(self.m_layers[0])
        #print(self.m_layers[0].get_nodes_value())\

        self.m_layers.append(Layer.hidden_layer_init(self.m_layers[0], 2))
        print(self.m_layers[1])

        self.m_layers.append(Layer.output_layer_init(self.m_layers[1], 1, Functions.init_cross_entropy()))
        print(self.m_layers[2].get_nodes_value())
        #print(self.m_layers[1].get_nodes_value())
        # number_of_weights_vector = layers[:-1]
        # number_of_features = self.m_X.shape[1]
        # number_of_weights_vector = np.concatenate((np.array([number_of_features]), number_of_weights_vector))
        #
        #
        # vNodes = np.vectorize(NetworkNode.generic_init)
        # print(number_of_weights_vector)
        #
        # for i in range(len(layers)):
        #
        #
        #     init_arr = vNodes(np.ones(layers[i], dtype=int) * number_of_weights_vector[i], )
        #     print(init_arr)
        #     self.m_layers.append(init_arr)
        #
        #     print(self.m_layers)

    def update_values(self, layer):
        pass

    def get_activation_value(self):
        return self.m_layers[-1].get_nodes_value()

    def get_accuracy(self, y, threshold=0.5):
        true_answers = y

        answers_list = self.get_activation_value()
        #print('actual:' + str(answers_list))
        answers_list = (threshold < answers_list) * 1.0

        true_predictions = y == answers_list

        # print(true_predictions)
        return 100 * np.count_nonzero(true_predictions) / true_predictions.shape[0]

    def fit(self, X_train, y_train, X_test, y_test):


        for i in range(10):

            print('fit predications :' + str(self.m_layers[0].get_nodes_value()))

            for layer in self.m_layers:
                layer.update_values()

            self.m_layers.reverse()
            print('predications :' + str(self.m_layers[0].get_nodes_value()))
            print(self.m_layers[0])
            print(y_train)
            first_grad = self.m_loss_func.der(self.m_layers[0].get_nodes_value(), y_train)

            print('grad: ' + str(first_grad))
            # for layer in self.m_layers[:-1]:
            #     weights_grad = layer.get_weights_gradient()
            #     layer.u
            gradient = self.m_layers[0].get_weights_gradient()
            print('sup:' + str(gradient))
            print(gradient.shape)
            print(first_grad.shape)
            first_grad = np.matmul(gradient, first_grad)
            print('weights grad' + str(first_grad))

            tst = self.m_layers[0].update_weights(first_grad)
            print('testt')
            print(tst.shape)
            print(tst)
            second_gradient = self.m_layers[1].get_weights_gradient()
            print(self.m_layers[1])
            print('sec:' + str(second_gradient.shape))
            print('first grad' + str(first_grad.shape))

            self.m_layers[1].update_weights(first_grad * second_gradient)

            self.m_layers.reverse()

            print('accuracy: ' + str(self.get_accuracy(y_train)))

            # loss_der = self.m_loss_func.der(X_train, y_train)
            #
            # for node in self.m_layers[0]:
            #     node.gradient_step()\
            #
            # for layer in self.m_layers:
            #     pass
            #     #if i < 3 or i % 100 == 0:



    def __repr__(self):
        layers_string = ""
        for layer in self.m_layers:
            layers_string += str(layer.get_length()) + ', '
        return f"NeuralNetwork({len(self.m_layers)} layers= [{layers_string}])"

