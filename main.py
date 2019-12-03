from convolution import Convolution2D
from pooling import MaxPool2D
from fully_connected import Softmax

import numpy as np
from tensorflow.keras.datasets import mnist

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# define the model
convolution = Convolution2D(filter_shape=(3,3), num_filters=8)
maxpool = MaxPool2D()
softmax = Softmax(13*13*8, 10)

def forward_propagation(img, label):
    output = convolution.conv2d(img/255)
    output = maxpool.pool(output)
    output = softmax.flatten(output)
    output = softmax.dense(output)
    
    loss = -np.log(output[label]) # -log(x) --> softmax loss function
    acc = 1 if np.argmax(output) == label else 0 # increase the accuracy if the predicted label = actual label
    
    return output, loss, acc

total_loss = 0
accuracy = 0

for i, (img, label) in enumerate(zip(x_train[:1000], y_train[:1000])): # let's train first 1000 data for simplicity 
    _, loss, acc = forward_propagation(img, label)
    total_loss += loss
    accuracy += acc
    
    if i % 100 == 0:
        print("Epoch", i, ": Loss= ", total_loss/100, "| Accuracy=", accuracy, "%")
        
        total_loss = 0
        accuracy = 0