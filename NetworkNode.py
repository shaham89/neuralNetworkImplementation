import math

import numpy as np

import helperFunctions
from Functions import Functions


class NetworkNode:

    """A node has a vector of inputs, weights plus a bias. And an activation function"""

    def __init__(self, weights, bias, input_values,
                         activation_func=Functions.init_sigmoid(),
                         loss_func=Functions.init_cross_entropy()):

        self.m_weights = weights
        self.m_bias = bias
        self.m_input_values = input_values
        self.activation_func = activation_func
        DEFAULT_LEARNING_RATE = 0.05
        self.weights_learning_rates = np.array([DEFAULT_LEARNING_RATE] * len(weights))
        self.bias_learning_rate = np.array([DEFAULT_LEARNING_RATE])
        self.m_loss_func = loss_func

    @classmethod
    def init_with_values(self, weights, bias, input_values,
                 activation_func=Functions.init_sigmoid(),
                 loss_func=Functions.init_cross_entropy()):

        self.m_weights = weights
        self.m_bias = bias
        self.m_input_values = input_values
        self.activation_func = activation_func
        DEFAULT_LEARNING_RATE = 0.05
        self.weights_learning_rates = np.array([DEFAULT_LEARNING_RATE] * len(weights))
        self.bias_learning_rate = np.array([DEFAULT_LEARNING_RATE])
        self.m_loss_func = loss_func


    @classmethod
    def generic_init(cls, number_of_weights, input_values=np.array([1, 1, 1])):
        return cls(np.random.rand(number_of_weights), 0, input_values,
                   Functions.init_sigmoid(), Functions.init_cross_entropy())


    @classmethod
    def dataset_init(cls, X):
        return cls(np.random.rand(X.shape[1]), 0, X,
                Functions.init_sigmoid(), Functions.init_cross_entropy())

    """Get the dot product plus the bias. 
    Example: bias = -3, weights= [1, 2, 5], X = [ [2, 1, 2], [8, 0, -5], [4, 4, 4] ]
    X is always 2 dimensional with axis1.len == weights.len
    """
    def dot_product(self, X=None):
        if X is None:
            X = self.m_input_values

        """The max value is set 15 so that """
        MAX_VALUE = 15
        max_array = np.ones(X.shape[0], dtype=float) * 1 * MAX_VALUE  # [MAX_VALUE, MAX_VALUE ... , MAX_VALUE]
        min_array = -max_array  # [-MAX_VALUE, -MAX_VALUE ... , -MAX_VALUE]

        return np.minimum(np.maximum(np.sum(self.m_weights * X, axis=1) + self.m_bias, min_array), max_array)


    def get_activation_value(self, X=None, dot_product=None):
        if X is None:
            X = self.m_input_values
        if dot_product is None:
            dot_product = self.dot_product(X)
        return self.activation_func.func(dot_product)



    def get_weights_gradient(self, X, y):

        dot = self.dot_product(X)
        #print("dot: "  + str(dot))
        #print("bias: " + str(self.m_bias))


        #print((np.sum(self.m_weights * X, axis=1) + self.m_bias).shape)
        #print((np.sum(self.m_weights * X, axis=1) + self.m_bias))

        act_der = self.activation_func.der(dot)
        #print("act_der: " + str(act_der))
        activation = self.get_activation_value(X)
        #print('activation: ' + str(activation))

        #   loss_der = self.m_loss_func.der(activation, y) #* act_der * self.m_weights
        #print('loss der: ' + str(loss_der))

        #print("X.T:" + str(-X.T))
        #avg_loss_der = loss_der * act_der
        avg_loss_der = act_der
        gradient = np.mean(avg_loss_der * X.T, axis=1)

        return gradient


    # only one derivative
    def get_bias_gradient(self, X, y):
        # y = mx + b, the output which is later passed to the activation function
        # output_value = self.dot_product(X)
        # print("bias")
        # print(output_value)
        # activated_value = self.activation_func.der(output_value)
        #
        # return self.m_loss_func.der(output_value, y) * activated_value

        dot = self.dot_product(X)
        #print("dot: " + str(dot))
        # print("bias: " + str(self.m_bias))

        # print((np.sum(self.m_weights * X, axis=1) + self.m_bias).shape)
        # print((np.sum(self.m_weights * X, axis=1) + self.m_bias))

        act_der = self.activation_func.der(dot)
        #print("act_der: " + str(act_der))
        activation = self.get_activation_value(X)
        #print('activation: ' + str(activation))

        #   loss_der = self.m_loss_func.der(activation, y)  # * act_der * self.m_weights
        #print('loss der: ' + str(loss_der))

        #print("X.T:" + str(-X.T))
        #   avg_loss_der = loss_der * act_der
        avg_loss_der = act_der
        #print("sum:" + str(avg_loss_der))
        gradient = np.mean(avg_loss_der * 1)
        #print('gradient:' + str(gradient))

        return gradient


    def update_parameters(self, X, y):
        self.m_input_values = X
        weights_gradient, bias_gradient = self.get_weights_gradient(X, y), self.get_bias_gradient(X, y)
        self.m_weights -= weights_gradient * self.weights_learning_rates
        #print('der:' + str(weights_gradient[0] * self.weights_learning_rates[0]))
        #print('weight:' + str(self.m_weights[-1]))
        self.m_bias -= bias_gradient * self.bias_learning_rate

    def gradient_step(self, X, y, weights_gradient, bias_der,  upper_layer_der):
        self.m_weights -= upper_layer_der * weights_gradient * self.weights_learning_rates
        # print('der:' + str(weights_gradient[0] * self.weights_learning_rates[0]))
        # print('weight:' + str(self.m_weights[-1]))
        self.m_bias -= bias_der * self.bias_learning_rate


    def fit(self, X_train, y_train, X_test, y_test):
        max_accuracy = 0
        for i in range(1000):


            if i < 3 or i % 100 == 0:
                self.print_stats(X_train, y_train, X_test, y_test)
            acc = self.get_accuracy(X_test, y_test)
            if max_accuracy < acc:
                max_accuracy = acc

            self.update_parameters(X_train, y_train)
        print('max:' + str(max_accuracy))

    def print_stats(self, X_train, y_train, X_test, y_test):
        print(self)
        activated_value = self.get_activation_value(X_train)
        # print(activated_value)
        print('-----------------------\navg loss: ' + str(
            np.average(self.m_loss_func.func(activated_value, y_train))) + '\n----------------')
        print(self.get_accuracy(X_train, y_train, 0.5))
        print(self.get_accuracy(X_test, y_test))


    def get_accuracy(self, X, y, threshold=0.5):
        true_answers = y

        answers_list = self.get_activation_value(X)

        answers_list = (threshold < answers_list) * 1.0

        true_predictions = y == answers_list

        #print(true_predictions)
        return 100 * np.count_nonzero(true_predictions) / true_predictions.shape[0]

    def __repr__(self):

        return f"NetworkNode({len(self.m_weights)} weights={self.m_weights}, bias={self.m_bias})"
