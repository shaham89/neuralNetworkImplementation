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
        DEFAULT_LEARNING_RATE = 0.0005
        self.weights_learning_rates = np.array([DEFAULT_LEARNING_RATE] * len(weights))
        self.bias_learning_rate = np.array([DEFAULT_LEARNING_RATE])
        self.m_loss_func = loss_func


    @classmethod
    def generic_init(cls, number_of_weights, input_values=np.array([1, 1, 1])):
        return cls(np.ones(number_of_weights), 0, input_values,
                   Functions.init_sigmoid(), Functions.init_cross_entropy())

        # self.m_weights =  np.ones(number_of_weights)
        # self.m_bias = 0
        # self.m_input_values = input_values
        # self.activation_func = helperFunctions.sigmoid_func

    #get the dot product plus the bias
    def calc_output(self, input_values=None):
        if input_values is None:
            input_values = self.m_input_values
        return np.dot(self.m_weights, input_values) + self.m_bias

    def get_activation_value(self, input_values=None):
        if input_values is None:
            input_values = self.m_input_values
        return self.activation_func.func(self.calc_output(input_values))

    def get_weights_gradient(self, y):
        # y = mx + b, the output which is later passed to the activation function
        output_value = self.calc_output()
        activated_value_der = self.activation_func.der(output_value)
        return self.m_loss_func.der(output_value, y) * activated_value_der * self.m_weights

    # only one derivative
    def get_bias_gradient(self, y):
        # y = mx + b, the output which is later passed to the activation function
        output_value = self.calc_output()
        activated_value = self.activation_func.der(output_value)

        return self.m_loss_func.der(output_value, y) * activated_value


    def update_parameters(self, X, y):
        self.m_input_values = X
        weights_gradient, bias_gradient = self.get_weights_gradient(y), self.get_bias_gradient(y)
        self.m_weights -= weights_gradient * self.weights_learning_rates
        self.m_bias -= bias_gradient * self.bias_learning_rate


    def __repr__(self):
        return f"NetworkNode(weights={self.m_weights}, bias={self.m_bias})"