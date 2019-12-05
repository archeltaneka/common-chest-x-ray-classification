import numpy as np

class Softmax:
    # initialize random weights and zero biases
    def __init__(self, num_features, num_nodes, activation):
        self.weights = np.random.randn(num_features, num_nodes) / num_features
        self.biases = np.zeros(num_nodes)
        self.activation = activation
    
    # connects flattened layer with a fully connected layer (dense)
    def dense(self, inputs):
        self.last_input_shape = inputs.shape # cache the last input shape BEFORE FLATTENING
        
        inputs = inputs.flatten() # flatten
        self.last_input = inputs # cache the last input shape AFTER FLATTENING
        input_features, nodes = self.weights.shape
        
        z = np.dot(inputs, self.weights) + self.biases # z = W . X + b
        self.z = z # cache z for backpropagation
        
        if(self.activation.lower() == 'softmax'):    
            a = np.exp(z) # a = g(z)
            return a / np.sum(a, axis=0) # e^a / sum(a)
        
        elif(self.activation.lower() == 'relu'):
            return max(0, z)
    
    # forward propagate phase
    def forward_propagation(self, img, label, output, reg_lambda):
        # -log(x) --> softmax loss function
        loss = (-np.log(output[label])) + (1/2 * reg_lambda * np.sum(self.weights ** 2)) # + regularization term
        acc = 1 if np.argmax(output) == label else 0 # increase the accuracy if the predicted label = actual label

        return output, loss, acc
    
    def back_propagation(self, dL, learning_rate, reg_lambda):
        for i, grad in enumerate(dL):
            if grad == 0: continue; # ignores 0 gradient
            
            exp_total = np.exp(self.z) # total of e^
            exp_sum = np.sum(exp_total) # sum of e^
            
            # gradients of z against totals
            dz = -exp_total[i] * exp_total / (exp_sum ** 2)
            dz[i] = exp_total[i] * (exp_sum - exp_total[i]) / (exp_sum ** 2)
            
            # gradients of totals against weights, biases, inputs
            dt_dw = self.last_input
            dt_db = 1
            dt_di = self.weights
            
            # gradients of loss against totals
            dL_dt = grad * dz
            
            # gradients of loss against weights, biases, and inputs
            dL_dw = np.dot(dt_dw[np.newaxis].T, dL_dt[np.newaxis])
            dL_db = dL_dt * dt_db
            dL_di = np.dot(dt_di, dL_dt)
            
            # add the regularization term
            dL_dw += reg_lambda * self.weights
            
            # update weights and biases
            self.weights -= learning_rate * dL_dw
            self.biases -= learning_rate * dL_db
            
            return dL_di.reshape(self.last_input_shape)