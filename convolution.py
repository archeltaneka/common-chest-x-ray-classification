import numpy as np

class Convolution2D:
    def __init__(self, filter_shape, num_filters):
        self.filter_shape = filter_shape
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, filter_shape[0], filter_shape[0]) # random initialization
        
    def iterate(self, img, filter_shape):
        height, width = img.shape
        
        for i in range(height - 2):
            for j in range(width - 2):
                output = img[i:(i+filter_shape[0]), j:(j+filter_shape[0])]
                yield output, i, j # 'yield' keyword will return any values and continue from the last value returned
    
    def conv2d(self, inputs):
        height, width = inputs.shape
        output = np.zeros((height-2, width-2, self.num_filters)) 
        
        for region, i, j in self.iterate(inputs, self.filter_shape):
            output[i, j] = np.sum(region * self.filters, axis=(1,2))
        
        return output