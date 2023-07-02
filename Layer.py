import numpy as np

from NetworkNode import *


class Layer:
    DEFAULT_LAYER = np.array([2, 1])

    def __init__(self, nodes,
                 pre_layer,
                 loss_function,
                 X,
                 layer_type,
                 is_constant):
        self.m_nodes = nodes
        self.m_pre_layer = pre_layer
        self.m_loss_function = loss_function
        self.m_X = X
        self.is_constant = is_constant
        self.number_of_nodes = len(nodes)
        self.m_type = layer_type

    @classmethod
    def input_layer_init(cls, X):

        # print(ac)
        # print(ac[0].get_activation_value())
        # print(NetworkNode.input_node_init(X.T))
        # print(ac(X.T[:]).shape)
        return cls(nodes=np.apply_along_axis(NetworkNode.input_node_init, 1, X.T[:]),
                   pre_layer=None,
                   loss_function=None,
                   X=X,
                   is_constant=True,
                   layer_type=TypeEnum.Types.INPUT_LAYER_TYPE)


    @classmethod
    def output_layer_init(cls, pre_layer, number_of_nodes, loss_function):
        nodes = np.ndarray(number_of_nodes, dtype=object)
        for i in range(nodes.shape[0]):
            nodes[i] = NetworkNode.output_node_init(pre_layer, loss_function)

        return cls(nodes=nodes,
                   pre_layer=pre_layer,
                   loss_function=loss_function,
                   X=None,
                   is_constant=False,
                   layer_type=TypeEnum.Types.OUTPUT_LAYER_TYPE)

    @classmethod
    def hidden_layer_init(cls, pre_layer, number_of_nodes):

        nodes = np.ndarray(number_of_nodes, dtype=object)
        for i in range(nodes.shape[0]):
            nodes[i] = NetworkNode.hidden_node_init(pre_layer)
        return cls(
            nodes=nodes,
            pre_layer=pre_layer,
            loss_function=None,
            X=None,
            is_constant=False,
            layer_type=TypeEnum.Types.HIDDEN_LAYER_TYPE)

    def get_nodes_value(self):
        if self.m_type == TypeEnum.Types.INPUT_LAYER_TYPE:
            return self.m_X
        #print([NetworkNode.get_value(node) for node in self.m_nodes[:7]])
        #print(np.column_stack([NetworkNode.get_value(node) for node in self.m_nodes]).shape)

        #print(np.apply_along_axis(NetworkNode.get_value, 0, self.m_nodes))
        return np.column_stack([NetworkNode.get_value(node) for node in self.m_nodes])

    def update_values(self):
        return np.vectorize(NetworkNode.update_value)(self.m_nodes)

    def update_weights(self, weights_grad):
        for node in self.m_nodes:
            node.update_weights(weights_grad)
        # np.vectorize()(self.m_nodes, weights_grad)

    def get_weights_gradient(self):
        return np.column_stack([node.get_weights_gradient() for node in self.m_nodes])

    def get_length(self):
        return self.number_of_nodes

    def __repr__(self):
        nodes_str = ''
        for node in self.m_nodes:
            nodes_str += str(node) + ', '
        return f"Layer({len(self.m_nodes)}, nodes={nodes_str})"

