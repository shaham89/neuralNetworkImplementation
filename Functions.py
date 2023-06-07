import numpy as np
import math

class Functions:

    def __init__(self, func, der):
        self.func = func
        self.der = der

    @staticmethod
    # sigmoid functions, but with very large or small value it rounds to the MIN,MAX values set
    def sigmoid_func(z):
        # MAX_VALUE = 709 #Crossing this value will cause a float overflow

        MIN_VALUE = np.ones(z.shape[0], dtype=float) * 1 * math.pow(10, -4)  # 0.000...1
        MAX_VALUE =  1 - MIN_VALUE  # 0.999...9

        output_val = np.minimum(np.maximum(1.0 / (1 + np.exp(-z)), MIN_VALUE), MAX_VALUE)  # actual sigmoid

        # if output_val < MIN_VALUE:
        #     return MIN_VALUE
        # elif output_val > MAX_VALUE:
        #     return MAX_VALUE

        return output_val

    @staticmethod
    def sigmoid_derivative(z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    @classmethod
    def init_sigmoid(cls):

        return cls(Functions.sigmoid_func, Functions.sigmoid_derivative)

    @staticmethod
    def log_loss_func(prediction, y):
        # prediction = min(prediction, 0.999)
        # prediction = max(prediction, 0.001)

        return -np.log(prediction) * y - np.log(1 - prediction) * (1 - y)

    @staticmethod
    def log_loss_derivative(predications, y):
        MIN_VALUE = np.ones(predications.shape[0], dtype=float) * 1 * math.pow(10, -4)  # 0.000...1
        MAX_VALUE = 1 - MIN_VALUE  # 0.999...9



        return -y / predications - (1 - y) / (predications - 1)

    @classmethod
    def init_cross_entropy(cls):
        return cls(Functions.log_loss_func, Functions.log_loss_derivative)



