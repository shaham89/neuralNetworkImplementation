import numpy as np

import helperFunctions


class NetworkNode:


    def __init__(self, weights, bias, input_values):
        self.m_weights = weights
        self.m_bias = bias
        self.m_input_values = input_values


    def __init__(self, number_of_weights, input_values):
        self.m_weights = np.ones(number_of_weights)
        self.m_bias = 0
        self.m_input_values = input_values

    def calc_logit_value(self):
        return helperFunctions.sigmoid_func(self.calc_output())

    def calc_output(self):
        return self.m_weights * self.m_input_values + self.m_bias



    def fit(self, actual_pred):
        act
