import numpy as np
import csv
from Functions import Functions
import helperFunctions
from NetworkNode import NetworkNode
from NeuralNetwork import NeuralNetwork


def m_converter(x):
    if x.decode() == '?':
        return 0
    return float(x)

"""#  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
  
  9. Class distribution:
 
   Benign: 458 (65.5%)
   Malignant: 241 (34.5%)
"""

# returns a numpy array of the data
def get_dataset_X_y():

    converters = {}
    for i in range(10):
        converters[i] = m_converter

    data = np.loadtxt("breast-cancer-wisconsin.data", delimiter=",",
                      dtype=float, converters=converters)

    MALIGNANT_VALUE = 4
    BENIGN_VALUE = 2

    malignant = data[data.T[-1] == MALIGNANT_VALUE] # [:3]
    benign = data[data.T[-1] == BENIGN_VALUE]
    benign = benign[:malignant.shape[0]]
    print(malignant.shape)
    print(benign.shape)
    data = np.concatenate((benign, malignant))

    y = (data.T[-1] / 2) - 1
    data.T[-1] = np.copy(y, subok=True) # convert the 2,4 into 0,1 for the classification
    data = np.delete(data, np.s_[-1], 1)  # remove the Y results

    X = np.delete(data, np.s_[0], 1)  # remove the ID

    print(X.shape)
    print(y.shape)

    return X, y

def main():

    X, y = get_dataset_X_y()


    node = NetworkNode.dataset_init(X)
    print(node)
    X_train, y_train, X_test, y_test = helperFunctions.HelperFunctions.train_test_split(X, y, 0.25)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    """NetworkNode(weights=[0.54201687 0.09545103 0.27370682 0.30382033 0.0181143  0.423715
 0.29786217 0.08282802 0.44836915], bias=[-8.657072])
"""
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
    nur = NeuralNetwork(X_train, y_train)
    return
    node.fit(X_train, y_train, X_test, y_test)
    print(node)
    print(node.get_accuracy(X_test, y_test))
    print(node.get_accuracy(X_train, y_train))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()