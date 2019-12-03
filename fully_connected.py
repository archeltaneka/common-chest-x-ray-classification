import numpy as np

class Softmax:
    # initialize random weights and zero biases
    def __init__(self, num_features, num_nodes):
        self.weights = np.random.randn(num_features, num_nodes) / num_features
        self.biases = np.zeros(num_nodes)
    
    # flattens out the previous layer
    def flatten(self, inputs):
        self.inputs = inputs.flatten()
    
    # connects flattened layer with a fully connected layer (dense)
    def dense(self, inputs):
        input_features, nodes = self.weights.shape
        
        z = np.dot(self.inputs, self.weights) + self.biases # z = W . X + b
        a = np.exp(z) # a = g(z)

        return a / np.sum(a, axis=0) # e^a / sum(a)