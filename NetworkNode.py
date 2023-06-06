import numpy as np

import helperFunctions
from Functions import Functions


class NetworkNode:

    """A node has a vector of inputs, weights plus a bias. And an activation function"""

    def __init__(self, weights, bias, input_values,
                 activation_func=Functions.init_sigmoid(),
                 loss_func=Functions.init_cross_entropy()):

        self.m_weights = weights
        self.m_bias = bias
        self.m_input_values = input_values
        self.activation_func = activation_func
        DEFAULT_LEARNING_RATE = 0.0005
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
        # return cls(np.concatenate((np.zeros(len(X[0]) - 3), np.random.rand(3))), 0, X,
        #            Functions.init_sigmoid(), Functions.init_cross_entropy())
        # self.m_weights =  np.ones(number_of_weights)
        # self.m_bias = 0
        # self.m_input_values = input_values
        # self.activation_func = helperFunctions.sigmoid_func

    #get the dot product plus the bias
    def dot_product(self, X=None):
        if X is None:
            X = self.m_input_values

        return np.sum(self.m_weights * X, axis=1) + self.m_bias

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
        print("dot: "  + str(dot))
        #print("bias: " + str(self.m_bias))


        #print((np.sum(self.m_weights * X, axis=1) + self.m_bias).shape)
        #print((np.sum(self.m_weights * X, axis=1) + self.m_bias))

        act_der = self.activation_func.der(dot)
        print("act_der: " + str(act_der))
        activation = self.get_activation_value(X)
        #print('activation: ' + str(activation))

        loss_der = self.m_loss_func.der(activation, y) #* act_der * self.m_weights
        print('loss der: ' + str(loss_der))

        print("X:" + str(X))
        avg_loss_der = np.average(loss_der * act_der)
        print("sum:" + str(avg_loss_der))
        gradient = avg_loss_der * -self.m_weights
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
        output_value = self.dot_product(X)
        print("bias")
        print(output_value)
        activated_value = self.activation_func.der(output_value)

        return self.m_loss_func.der(output_value, y) * activated_value


    def update_parameters(self, X, y):
        self.m_input_values = X
        weights_gradient, bias_gradient = self.get_weights_gradient(X, y), 0 #self.get_bias_gradient(X, y)
        self.m_weights[0] -= weights_gradient[0] * self.weights_learning_rates[0]
        print('der:' + str(weights_gradient[0] * self.weights_learning_rates[0]))
        #print('weight:' + str(self.m_weights[-1]))
        self.m_bias -= bias_gradient * self.bias_learning_rate

    def fit(self, X, y):
        for i in range(10):


            if i < 10 or i % 10 == 0:
                #print("pred:" + str(self.get_activation_value()))
                #print(np.self.m_loss_func.func(self.get_activation_value(), y))
                print(self)
                activated_value = self.get_activation_value(X)
                # print(activated_value)
                print('avg loss: ' + str(np.average(self.m_loss_func.func(activated_value, y))))

            self.update_parameters(X, y)


    def __repr__(self):

        return f"NetworkNode(weights={self.m_weights}, bias={self.m_bias})"
