import math
import sys

import numpy as np


# sigmoid functions, but with very large or small value it rounds to the MIN,MAX values set
def sigmoid_func(z):
    # MAX_VALUE = 709 #Crossing this value will cause a float overflow

    MIN_VALUE = 1 * math.pow(10, -10)      # 0.000...1
    MAX_VALUE = 1 - MIN_VALUE              # 0.999...9
    output_val = (1.0 / (1 + math.exp(-z)))  # actual sigmoid

    if output_val < MIN_VALUE:
        return MIN_VALUE
    elif output_val > MAX_VALUE:
        return MAX_VALUE

    return output_val

def sigmoid_derivative(z):
    return np.exp(-z) - 1 / ((1 + np.exp(-z)) ** 2)

def log_loss_func(prediction, y):
    return np.log(prediction) * y + np.log(1 - prediction) * (1 - y)

def log_loss_derivative(pred, y):
    return y / pred + (1 - y) / (pred - 1)



# pred = activation(wx + b)
# loss(pred) = ln(pred) * y + ln(1 - pred) * (1 - y)
# sigmoid(wx + b)/dw = w *
# loss`(w) = loss`(pred) * pred`(w)

# loss`(pred) = y / pred + (1 - y) / (pred - 1)
# loss`(w) =  * loss`(pred)
# loss`(b) = loss`(pred)


