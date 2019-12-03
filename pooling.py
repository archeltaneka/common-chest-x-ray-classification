import numpy as np

class MaxPool2D:
    def iterate(self, img):
        height, width, _ = img.shape
        h = height//2
        w = width//2
        
        for i in range(h):
            for j in range(w):
                new_region = img[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield new_region, i, j
    
    def pool(self, inputs):
        height, width, num_filters = inputs.shape
        output = np.zeros((height//2, width//2, num_filters))
        
        for img_region, i, j in self.iterate(inputs):
            output[i, j] = np.max(img_region, axis=(0,1))
        
        return output