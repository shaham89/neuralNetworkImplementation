from NetworkNode import *


class Layer:
    DEFAULT_LAYER = np.array([2, 1])

    def __init__(self, nodes, pre_layer, loss_function, X, is_constant):
        self.m_nodes = nodes
        self.m_pre_layer = pre_layer
        self.m_loss_function = loss_function
        self.m_X = X
        self.is_constant = is_constant
        self.number_of_nodes = len(nodes)


    @classmethod
    def input_layer_init(cls, X, number_of_nodes):

        return cls(nodes=np.vectorize(NetworkNode.input_layer_node_init)(np.ones(number_of_nodes, dtype=int)),
                   pre_layer=None,
                   loss_function=None,
                   X=X,
                   is_constant=True)


    @classmethod
    def output_layer_init(cls, pre_layer, number_of_nodes, loss_function):
        return cls(nodes=np.vectorize(NetworkNode.hidden_layer_node_init)(np.full(number_of_nodes, pre_layer.get_length(), dtype=int)),
                   pre_layer=pre_layer,
                   loss_function=loss_function,
                   X=None,
                   is_constant=False)

    @classmethod
    def hidden_layer_init(cls, pre_layer, number_of_nodes):
        return cls(
            nodes=np.vectorize(NetworkNode.hidden_layer_node_init)(np.full(number_of_nodes, pre_layer.get_length(), dtype=int)),
            pre_layer=pre_layer,
            loss_function=None,
            X=None,
            is_constant=False)

    def get_nodes_value(self):
        return np.vectorize(NetworkNode.get_value)(self.m_nodes)

    def update_values(self):
        return np.vectorize(NetworkNode.update_value)(self.m_nodes)


    def get_length(self):
        return self.number_of_nodes


