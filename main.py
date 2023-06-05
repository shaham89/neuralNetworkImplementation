import numpy as np

from Functions import Functions
import helperFunctions
from NetworkNode import NetworkNode


def main():
    weights = np.array([1, 0.5, -1])
    bias = 4
    inputs = np.array([1, 1, 1])

    node = NetworkNode.generic_init(1, 6)  # NetworkNode(weights, bias, inputs)

    print(node)

    features_outputs = list()
    for i in range(16):
        if i > 7:
            features_outputs.append((i, 1))
        else:
            features_outputs.append((i, 0))

    print(features_outputs)
    print(node.m_loss_func.func(0.7, 1))


    for i in range(100):
        for j in range(16):
            X, y = features_outputs[j][0], features_outputs[j][1]
            node.update_parameters(X, y)
            print("pred:" + str(node.get_activation_value()))
            print(node.m_loss_func.func(node.get_activation_value(), y))
            print(node)
    print(node.calc_output(9))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

