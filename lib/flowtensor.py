import numpy as np
from lib.utils import padding

class Convolution2D:
    '''
        Convolutional 2D layer
        
        This class will create a convolutional layer, convolved with an image
        
        # Arguments
            filter_shape: tuple/list of 2 integers. This specifies the height and width of the kernel (convolution window)
            num_filters: Integer, the number of filters (depth) in the convolution
            padding: 'same' padding will retain the image's shape, 'valid' does not retain image shape (image's shape
                      will shrink after convolution)
            stride: Integer, specifies the number of strides of the convolution
            activation: The kernel activation function. As of this time, 'relu' is the only activation function.
            kernel_init: Initializers for kernel weights. Use 'random' or 'xavier', 'random' by default.
            debugging: Boolean. Setting this to True will enable debugging feature, False by default.
            
        # Input
            2D object with shape:
            `(height, width)`
            
        # Output
            3D object with shape:
            `(new_height, new_width, filters)` 
    '''
    
    def __init__(self, filter_shape, num_filters, padding, stride, activation, kernel_init='random', debugging='False'):
        self.filter_shape = filter_shape
        self.num_filters = num_filters
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self.kernel_init = kernel_init
        self.debugging = debugging
        
        # random kernel initialization
        if self.kernel_init.lower() == 'random':
            self.filters = np.random.randn(num_filters, filter_shape[0], filter_shape[0])
        # xavier kernel initialization a.k.a glorot normal
        elif self.kernel_init.lower() == 'xavier':
            self.filters = np.random.randn(num_filters, filter_shape[0], filter_shape[0]) / 9 # 1 / N
    
    # convolving through the entire image pixels with the filter
    def iterate(self, img, filter_shape):
        height, width = img.shape
        height = int((height + 2 * self.pad_size - self.filter_shape[0]) / self.stride) + 1
        width = int((width + 2 * self.pad_size - self.filter_shape[0]) / self.stride) + 1
        # check for odd heights and widths, then round the height and width to the nearest integer
        if height % 2 != 0 and width % 2 != 0:
            height += 1
            width += 1
        
        for i in range(height-(filter_shape[0]-1)):
            for j in range(width-(filter_shape[0]-1)):
                output = img[i*self.stride:(i*self.stride+filter_shape[0]), j*self.stride:(j*self.stride+filter_shape[0])]
                yield output, i, j # 'yield' keyword will return any values and continue from the last value returned
    
    # convolutional operation which is similar with the matrix multiplication
    def conv2d(self, inputs):
        if self.debugging==True: print("Image before padding:", inputs.shape)
        self.last_input = inputs # cache the last input for backpropagation
    
        # padding
        if(self.padding.lower() == 'same'): # same padding
            height, width = inputs.shape
            
            pad_size = int(((height * self.stride) - height + self.filter_shape[0] - 1) / 2)
            self.pad_size = pad_size

            inputs = padding(inputs, pad_size) # apply padding according to the pad_size
            height, width = inputs.shape # reinitialize height and width with padded image

            new_height = int((height + 2 * pad_size - self.filter_shape[0]) / self.stride) + 1
            new_width = int((width + 2 * pad_size - self.filter_shape[0]) / self.stride) + 1
            # check for odd heights and widths
            if new_height % 2 != 0 and new_width % 2 != 0:
                new_height += 1
                new_width += 1
                
            output = np.zeros((new_height, new_width, self.num_filters))
            
        elif(self.padding.lower() == 'valid'): # valid/no padding
            height, width = inputs.shape
            self.pad_size = 0
            output = np.zeros((height-(self.filter_shape[0]-1), width-(self.filter_shape[0]-1), self.num_filters))
            
        if self.debugging==True: print("Image after padding:", inputs.shape)
        
        for region, i, j in self.iterate(inputs, self.filter_shape):
            output[i, j] = np.sum(region * self.filters, axis=(1,2))
            if self.activation.lower() == 'relu':
                output[i, j] = np.maximum(0, output[i, j])
        
        if self.debugging==True:print("Total parameters after convolution: ", output.shape, "=", output.shape[0]*output.shape[1]*output.shape[2])
            
        return output
    
    # backpropagation step
    def back_propagation(self, dL, learning_rate):
        dL_filters = np.zeros(self.filters.shape)
        
        for img_region, i, j in self.iterate(self.last_input, self.filter_shape):
            for k in range(self.num_filters):
                dL_filters[k] += dL[i, j, k] * img_region
        
        # don't forget to update the filters
        self.filters -= learning_rate * dL_filters
        
        return None


    
class MaxPool2D:
    '''
        MaxPool 2D layer
        
        This class will create a maxpool layer, which takes the biggest (maximum) value from the input
        
        # Arguments
            pool_size: Integer. Specifies the pooling scale. e.g: pool_size=2 will halve the input dimensions. 
            debugging: Boolean. Setting this to True will enable debugging feature, False by default.
            
        # Input
            3D object with shape:
            `(height, width, filters)`
            
        # Output
            3D object with shape:
            `(new_height, new_width, filters)` 
    '''
    
    def __init__(self, pool_size, debugging=False):
        self.pool_size = pool_size
        self.debugging = debugging
    
    # iterate through the entire input to get the maximum value
    def iterate(self, img):
        height, width, _ = img.shape
        h = height//self.pool_size
        w = width//self.pool_size
        
        for i in range(h):
            for j in range(w):
                new_region = img[(i*self.pool_size):(i*self.pool_size+self.pool_size), 
                                 (j*self.pool_size):(j*self.pool_size+self.pool_size)]
                yield new_region, i, j
    
    def pool(self, inputs):
        self.last_input = inputs # cache the last input for backpropagation
        
        height, width, num_filters = inputs.shape
        output = np.zeros((height//self.pool_size, width//self.pool_size, num_filters))
        
        for img_region, i, j in self.iterate(inputs):
            output[i, j] = np.max(img_region, axis=(0,1)) # np.max to retrieve the biggest (maximum) value
        
        if self.debugging==True: print("Dimension after maxpooling:", output.shape)
        
        return output
    
    # backpropagation step
    def back_propagation(self, dL_output):
        dL_input = np.zeros((self.last_input.shape))
        
        for img_region, i, j in self.iterate(self.last_input):
            height, width, num_filters = img_region.shape
            # find the max value for each region
            maxi = np.max(img_region, axis=(0,1))
            
            for k in range(height):
                for l in range(width):
                    for m in range(num_filters):
                        if img_region[k, l, m] == maxi[m]: # if the max values match, copy the gradient
                            dL_input[i*2+k, j*2+l, m] = dL_output[i, j, m]
        return dL_input

    
    
class Softmax:
    '''
        Softmax layer
        
        This class will flatten and connect input layer with a dense connected layer
        
        # Arguments
            num_features: Integer. Specifies the last layer's number of features
            num_nodes: Integer. Specifies the number of nodes in the layer (the number of classes in the dataset).
            activation: Specifies the activation function. 'softmax' or 'relu'
            debugging: Boolean. Setting this to True will enable debugging feature, False by default.
            
        # Input
            3D object with shape:
            `(height, width, filters)`
            
        # Output
            2D object with shape:
            `(num_features, predictions)` 
    '''
    
    # initialize random weights and zero biases
    def __init__(self, num_features, num_nodes, activation, regularizer='none', debugging='False'):
        self.weights = np.random.randn(num_features, num_nodes) / num_features
        self.biases = np.zeros(num_nodes)
        self.activation = activation
        self.regularizer = regularizer
        if debugging==True: print("Total parameters to train:", num_features, "x", num_nodes, "=", num_features * num_nodes)
    
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
    def forward_propagation(self, img, label, output, reg_lambda=1e-2):
        # -log(x) --> softmax loss function
        loss = (-np.log(output[label]))
        
        # regularizer
        if self.regularizer.lower() == 'l2':
            loss = loss + (1/2 * reg_lambda * np.sum(self.weights ** 2)) # + (1/2) * reg_lambda * (w^2) --> regularization term
        
        acc = 1 if np.argmax(output) == label else 0 # increase the accuracy if the predicted label = actual label
        
        return output, loss, acc
    
    # backpropagation stetp
    def back_propagation(self, dL, learning_rate, reg_lambda=1e-2):
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