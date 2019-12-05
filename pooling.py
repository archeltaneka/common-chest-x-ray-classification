import numpy as np

class MaxPool2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
    
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
            output[i, j] = np.max(img_region, axis=(0,1))
        
        return output
    
    # backpropagation
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