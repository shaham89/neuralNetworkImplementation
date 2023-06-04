import numpy as np

import helperFunctions
from NetworkNode import NetworkNode


def main():
    weights = np.array([1, 0.5, -1])
    bias = 4
    inputs = np.array([1, 1, 1])

    node = NetworkNode(weights, bias, inputs)

    print(node.calc_output())
    print(node.get_activation_value())
    print(node.get_weights_gradient())
    print(node)
    for i in range(1000):
        node.update_parameters(node.get_weights_gradient(), node.get_bias_gradient())
        print(node)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

