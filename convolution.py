import numpy as np
from utils import padding

class Convolution2D:
    def __init__(self, filter_shape, num_filters, padding, stride, activation='relu', kernel_init='None'):
        self.filter_shape = filter_shape
        self.num_filters = num_filters
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self.kernel_init = kernel_init
        
        # random kernel initialization
        if self.kernel_init.lower() == 'none':
            self.filters = np.random.randn(num_filters, filter_shape[0], filter_shape[0])
        # xavier kernel initialization a.k.a glorot normal
        elif self.kernel_init.lower() == 'xavier':
            self.filters = np.random.randn(num_filters, filter_shape[0], filter_shape[0]) / 9 # 1 / N
        
    def iterate(self, img, filter_shape):
        height, width = img.shape
        height = int((height + 2 * self.pad_size - self.filter_shape[0]) / self.stride) + 1
        width = int((width + 2 * self.pad_size - self.filter_shape[0]) / self.stride) + 1
        # check for odd heights and widths
        if height % 2 != 0 and width % 2 != 0:
            height += 1
            width += 1
#         print(height)
#         print(width)
#         print(filter_shape[0]-1)
        
        for i in range(height-(filter_shape[0]-1)):
            for j in range(width-(filter_shape[0]-1)):
                output = img[i*self.stride:(i*self.stride+filter_shape[0]), j*self.stride:(j*self.stride+filter_shape[0])]
#                 print(output, i, j)
                yield output, i, j # 'yield' keyword will return any values and continue from the last value returned
    
    def conv2d(self, inputs):
#         print("Before padding: ", inputs.shape)
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
            
#         print("After padding: ", inputs.shape)
        
        for region, i, j in self.iterate(inputs, self.filter_shape):
            output[i, j] = np.sum(region * self.filters, axis=(1,2))
            if self.activation.lower() == 'relu':
                output[i, j] = np.maximum(0, output[i, j])
        
#         print("After convolution: ", output.shape)
        return output

    def back_propagation(self, dL, learning_rate):
        dL_filters = np.zeros(self.filters.shape)
        
        for img_region, i, j in self.iterate(self.last_input, self.filter_shape):
            for k in range(self.num_filters):
                dL_filters[k] += dL[i, j, k] * img_region
        
        # don't forget to update the filters
        self.filters -= learning_rate * dL_filters
        
        return None