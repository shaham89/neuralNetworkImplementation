import numpy as np

import helperFunctions



class NetworkNode:

    """A node has a vector of inputs, weights plus a bias. And an activation function"""

    def __init__(self, weights, bias, input_values, activation_func=helperFunctions.sigmoid_func):
        self.m_weights = weights
        self.m_bias = bias
        self.m_input_values = input_values
        self.activation_func = activation_func
        DEFAULT_LEARNING_RATE = 0.5
        self.weights_learning_rates = np.array([DEFAULT_LEARNING_RATE] * len(weights))
        self.bias_learning_rates = np.array([DEFAULT_LEARNING_RATE])

    @classmethod
    def generic_init(cls, number_of_weights, input_values=np.array([1, 1, 1])):
        return cls(np.ones(number_of_weights), 0, input_values, helperFunctions.sigmoid_func)
        # self.m_weights =  np.ones(number_of_weights)
        # self.m_bias = 0
        # self.m_input_values = input_values
        # self.activation_func = helperFunctions.sigmoid_func

    def calc_output(self):
        return np.dot(self.m_weights, self.m_input_values) + self.m_bias

    def get_activation_value(self):
        return self.activation_func(self.calc_output())

    def get_weights_gradient(self):
        # y = mx + b, the output which is later passed to the activation function
        output_value = self.calc_output()

        return helperFunctions.sigmoid_derivative(output_value) * self.m_weights

    # only one derivative
    def get_bias_gradient(self):
        # y = mx + b, the output which is later passed to the activation function
        output_value = self.calc_output()

        return helperFunctions.sigmoid_derivative(output_value)

    def update_parameters(self, weights_gradient, bias_gradient):
        self.m_weights -= weights_gradient * self.weights_learning_rates
        self.m_bias -= bias_gradient * self.bias_learning_rates

    def __repr__(self):
        return f"NetworkNode(weights={self.m_weights}, bias={self.m_bias})"

