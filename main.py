import numpy as np
import csv
from Functions import Functions
import helperFunctions
from NetworkNode import NetworkNode

#returns a numpy array of the data
def get_dataset():
    data = np.loadtxt("breast-cancer-wisconsin.data",delimiter=",", dtype=str)

    data = np.delete(data, np.s_[0], 1)
    data = data.reshape((data.shape[0], 2, data.shape[1] - 1))
    print(data.shape)
    print(data)

def main():
    weights = np.array([1, 0.5, -1])
    bias = 4
    inputs = np.array([1, 1, 1])
    get_dataset()
    return
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


    for i in range(1000):
        for j in range(16):
            X, y = features_outputs[j][0], features_outputs[j][1]
            node.update_parameters(X, y)
        print("pred:" + str(node.get_activation_value()))
        print(node.m_loss_func.func(node.get_activation_value(), y))
        print(node)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()