import numpy as np
import csv
from Functions import Functions
import helperFunctions
from NetworkNode import NetworkNode

def m_converter(x):
    if x.decode() == '?':
        return 0
    return float(x)

#returns a numpy array of the data
def get_dataset_X_y():

    converters = {}
    for i in range(9):
        converters[i] = m_converter

    data = np.loadtxt("breast-cancer-wisconsin.data", delimiter=",",
                      dtype=float, converters=converters, skiprows=5 ,max_rows=2)

    y = (data.T[-1] / 2) - 1

    data.T[-1] = np.copy(y, subok=True) # convert the 2,4 into 0,1 for the classification

    X = np.delete(data, np.s_[0], 1)  # remove the ID
    #data = np.delete(data, np.s_[0], -1) # remove the Y results
    print(X.shape)
    print(y.shape)
    print(X)
    return X, y

def main():
    weights = np.array([1, 0.5, -1])
    bias = 4
    inputs = np.array([1, 1, 1])
    X, y = get_dataset_X_y()


    node = NetworkNode.dataset_init(X)
    print(node)
    # node = NetworkNode.generic_init(1, 6)  # NetworkNode(weights, bias, inputs)
    #
    # print(node)
    #
    # features_outputs = list()
    # for i in range(16):
    #     if i > 7:
    #         features_outputs.append((i, 1))
    #     else:
    #         features_outputs.append((i, 0))
    #
    # print(features_outputs)
    #print(node.m_loss_func.func(0.7, 1))


    # for i in range(1000):
    #     for j in range(16):
    #         X, y = features_outputs[j][0], features_outputs[j][1]
    #         node.update_parameters(X, y)
    #     print("pred:" + str(node.get_activation_value()))
    #     print(node.m_loss_func.func(node.get_activation_value(), y))
    #     print(node)

    node.fit(X, y)
    print(node)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()