import math

import numpy as np

import helperFunctions
from Functions import Functions


def plot(x1, y1, x2, y2, x_axis_title="x axis caption", y_axis_title="y axis caption", title='Title', islog=True):
    from matplotlib import pyplot as plt

    plt.title(title)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    if islog:
        plt.yscale('log')

    plt.plot(x1, y1, color="blue")
    plt.plot(x2, y2, color="red")

    plt.legend(['train', 'test'])

    plt.show()


class NetworkNode:

    """A node has a vector of inputs, weights plus a bias. And an activation function"""

    def __init__(self, weights, bias, input_values,
                 activation_func=Functions.init_sigmoid(),
                 loss_func=Functions.init_cross_entropy()):

        self.m_weights = weights
        self.m_bias = bias
        self.m_input_values = input_values
        self.activation_func = activation_func
        DEFAULT_LEARNING_RATE = 0.05
        self.weights_learning_rates = np.array([DEFAULT_LEARNING_RATE] * len(weights))
        self.bias_learning_rate = np.array([DEFAULT_LEARNING_RATE])
        self.m_loss_func = loss_func


    @classmethod
    def generic_init(cls, number_of_weights, input_values=np.array([1, 1, 1])):
        return cls(np.ones(number_of_weights), 0, input_values,
                   Functions.init_sigmoid(), Functions.init_cross_entropy())

        """NetworkNode(weights=[0.50235337 0.64351151 1.53891935 0.78830567 1.70083465 0.43906473
 1.86396637 0.66925708 0.18288869 0.93735947], bias=[0.])
 total loss: 2219.24635813194

"""
        # self.m_weights =  np.ones(number_of_weights)
        # self.m_bias = 0
        # self.m_input_values = input_values
        # self.activation_func = helperFunctions.sigmoid_func

    @classmethod
    def dataset_init(cls, X):
        return cls(np.random.rand(X.shape[1]), 0, X,
                Functions.init_sigmoid(), Functions.init_cross_entropy())
        #np.random.rand(1)
        #return cls(np.concatenate((np.zeros(len(X[0]) - 2), np.array([0.5, 2]))), 0, X,
        #           Functions.init_sigmoid(), Functions.init_cross_entropy())
        # self.m_weights =  np.ones(number_of_weights)
        # self.m_bias = 0
        # self.m_input_values = input_values
        # self.activation_func = helperFunctions.sigmoid_func

    #get the dot product plus the bias
    def dot_product(self, X=None):
        if X is None:
            X = self.m_input_values

        MAX_VALUE = np.ones(X.shape[0], dtype=float) * 1 * 15  # 0.000...1
        MIN_VALUE = -MAX_VALUE  # 0.999...9
        # print(X)
        # print(MIN_VALUE)
        # print(MAX_VALUE)
        # print(self.m_weights)
        # print(np.sum(self.m_weights * X, axis=1))
        #print(np.sum(self.m_weights * X, axis=1))

        #print(np.matmul(X, self.m_weights))

        #return np.minimum(np.maximum(np.sum(self.m_weights * X, axis=1) + self.m_bias, MIN_VALUE), MAX_VALUE)
        return np.minimum(np.maximum(np.matmul(X, self.m_weights) + self.m_bias, MIN_VALUE), MAX_VALUE)


    def get_activation_value(self, X=None, dot_product=None):
        if X is None:
            X = self.m_input_values
        if dot_product is None:
            dot_product = self.dot_product(X)
        return self.activation_func.func(dot_product)



    def get_weights_gradient(self, X, y):
        #print("testing")
        # y = mx + b, the output which is later passed to the activation function
        # print(X.shape)
        # print(self.m_weights.shape)
        # print((self.m_weights * X).shape)
        # print("weights: " + str(self.m_weights))
        # print("X:" + str(X))
        # print("mul" + str(self.m_weights * X))
        dot = self.dot_product(X)
        #print("dot: "  + str(dot))
        #print("bias: " + str(self.m_bias))


        #print((np.sum(self.m_weights * X, axis=1) + self.m_bias).shape)
        #print((np.sum(self.m_weights * X, axis=1) + self.m_bias))

        act_der = self.activation_func.der(dot)
        #print("act_der: " + str(act_der))
        activation = self.get_activation_value(X)
        #print('activation: ' + str(activation))

        loss_der = self.m_loss_func.der(activation, y) #* act_der * self.m_weights
        #print('loss der: ' + str(loss_der))

        #print("X.T:" + str(-X.T))
        avg_loss_der = loss_der * act_der
        # print("sum:" + str(avg_loss_der))
        # print('muli:' + str(avg_loss_der * -X.T))
        gradient = np.mean(avg_loss_der * X.T, axis=1)
        #print('gradient:' + str(gradient))

        return gradient

        # output_value = self.dot_product(X)
        # print("weights")
        # print(output_value.T)
        #
        #
        # activated_value_der = self.activation_func.der(output_value)
        #
        # print(self.m_loss_func.der(output_value, y) * activated_value_der * self.m_weights)
        # return np.sum(self.m_loss_func.der(output_value, y) * activated_value_der) * self.m_weights

    # only one derivative
    def get_bias_gradient(self, X, y):
        # y = mx + b, the output which is later passed to the activation function
        # output_value = self.dot_product(X)
        # print("bias")
        # print(output_value)
        # activated_value = self.activation_func.der(output_value)
        #
        # return self.m_loss_func.der(output_value, y) * activated_value

        dot = self.dot_product(X)
        #print("dot: " + str(dot))
        # print("bias: " + str(self.m_bias))

        # print((np.sum(self.m_weights * X, axis=1) + self.m_bias).shape)
        # print((np.sum(self.m_weights * X, axis=1) + self.m_bias))

        act_der = self.activation_func.der(dot)
        #print("act_der: " + str(act_der))
        activation = self.get_activation_value(X)
        #print('activation: ' + str(activation))

        loss_der = self.m_loss_func.der(activation, y)  # * act_der * self.m_weights
        #print('loss der: ' + str(loss_der))

        #print("X.T:" + str(-X.T))
        avg_loss_der = loss_der * act_der
        #print("sum:" + str(avg_loss_der))
        gradient = np.mean(avg_loss_der * 1)
        #print('gradient:' + str(gradient))

        return gradient

    def get_loss(self, X, y):
        return np.average(self.m_loss_func.func(self.get_activation_value(X), y))

    def update_parameters(self, X, y):
        self.m_input_values = X
        weights_gradient, bias_gradient = self.get_weights_gradient(X, y), self.get_bias_gradient(X, y)
        self.m_weights -= weights_gradient * self.weights_learning_rates
        #print('der:' + str(weights_gradient[0] * self.weights_learning_rates[0]))
        #print('weight:' + str(self.m_weights[-1]))
        self.m_bias -= bias_gradient * self.bias_learning_rate

    def fit(self, X_train, y_train, X_test, y_test):
        NUMBER_OF_EPOCHS = 200
        train_loss_array = np.zeros(NUMBER_OF_EPOCHS, dtype=np.float)
        test_loss_array = np.zeros(NUMBER_OF_EPOCHS, dtype=np.float)

        train_loss_accuracy = np.zeros(NUMBER_OF_EPOCHS, dtype=np.float)
        test_loss_accuracy = np.zeros(NUMBER_OF_EPOCHS, dtype=np.float)



        for i in range(NUMBER_OF_EPOCHS):


            train_loss_array[i] = self.get_loss(X_train, y_train)
            test_loss_array[i] = self.get_loss(X_test, y_test)

            train_loss_accuracy[i] = self.get_accuracy(X_train, y_train, threshold=0.64)
            test_loss_accuracy[i] = self.get_accuracy(X_test, y_test, threshold=0.64)

            if i < 3 or i % 100 == 0:
                #print("pred:" + str(self.get_activation_value()))
                #print(np.self.m_loss_func.func(self.get_activation_value(), y))
                print(self)
                # print(activated_value)
                print('-----------------------\navg loss: ' + str(self.get_loss(X_train, y_train)) + '\n----------------')
                print(self.get_accuracy(X_train, y_train, 0.5))
                print(self.get_accuracy(X_test, y_test))



            self.update_parameters(X_train, y_train)

        plot(range(NUMBER_OF_EPOCHS), train_loss_array, range(NUMBER_OF_EPOCHS), test_loss_array
             , 'epoch number', 'loss', 'Loss over iterations', True)

        print('----------------------------------')
        plot(range(NUMBER_OF_EPOCHS), train_loss_accuracy, range(NUMBER_OF_EPOCHS), test_loss_accuracy
             , 'epoch number', 'accuracy', 'accuracy over iterations', False)

        optimal_threshold, tpr, fpr = self.get_roc_curve_threshold(X_train, y_train)
        print('-----Optimal threshold: ' + str(optimal_threshold))

        plot(fpr, tpr, np.array(0), np.array(0), 'fpr', 'tpr', 'roc curve', False)

        print('-----training max accuracy: ' + str(self.get_accuracy(X_train, y_train, threshold=optimal_threshold)))
        print('-----testing accuracy: ' + str(self.get_accuracy(X_test, y_test, threshold=optimal_threshold)))
        print('----------------------------------')
        print(self.get_tpr(X_train, y_train, optimal_threshold))
        print(self.get_fpr(X_train, y_train, optimal_threshold))

    def get_roc_curve_threshold(self, X, y):
        threshold = 1
        max_accuracy = 0
        optimal_threshold = threshold

        tpr_rate = np.zeros(int(threshold / 0.01), dtype=np.float)
        fpr_rate = np.zeros(int(threshold / 0.01), dtype=np.float)

        i = 0
        while threshold > 0:
            curr_acc = self.get_accuracy(X, y, threshold)

            tpr_rate[i] = self.get_tpr(X, y, threshold)
            fpr_rate[i] = self.get_fpr(X, y, threshold)

            i += 1
            if curr_acc > max_accuracy:
                max_accuracy = curr_acc
                optimal_threshold = threshold

            threshold -= 0.01

        return optimal_threshold, tpr_rate, fpr_rate


    def get_tpr(self, X, y, threshold):
        answers_list = self.get_activation_value(X)

        answers_list = (threshold < answers_list) * 1.0

        tp = np.logical_and(answers_list == 1, y == 1)

        fn = np.logical_and(answers_list == 0, y == 1)

        return  np.count_nonzero(tp) / (np.count_nonzero(tp) + np.count_nonzero(fn))

    def get_fpr(self, X, y, threshold):
        answers_list = self.get_activation_value(X)

        answers_list = (threshold < answers_list) * 1.0

        tn = np.logical_and(answers_list == 0, y == 0)

        fp = np.logical_and(answers_list == 1, y == 0)

        return  np.count_nonzero(fp) / (np.count_nonzero(fp) + np.count_nonzero(tn))


    """return model accuracy as a percentage"""
    def get_accuracy(self, X, y, threshold=0.5):
        true_answers = y

        answers_list = self.get_activation_value(X)

        answers_list = (threshold < answers_list) * 1.0

        true_predictions = true_answers == answers_list

        #print(true_predictions)
        return 100 * np.count_nonzero(true_predictions) / true_predictions.shape[0]

    def __repr__(self):

        return f"NetworkNode(weights={self.m_weights}, bias={self.m_bias})"
