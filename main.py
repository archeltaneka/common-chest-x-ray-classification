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

# forward propagation phase
def forward_propagation(img, label):
    output = convolution.conv2d(img/255)
    output = maxpool.pool(output)
    output = softmax.dense(output)
    
    loss = -np.log(output[label]) # -log(x) --> softmax loss function
    acc = 1 if np.argmax(output) == label else 0 # increase the accuracy if the predicted label = actual label
    
    return output, loss, acc

# train the model
def train(img, label, learning_rate):
    # forward propagation
    output, loss, acc = forward_propagation(img, label)
    
    # initial gradient
    grad = np.zeros(10) # 10 different classes
    grad[label] = -1 / output[label]
    
    # back propagation
    grad = softmax.back_propagation(grad, learning_rate)
    
    return loss, acc

l = 0
acc = 0

for i, (img, label) in enumerate(zip(x_train[:1000], y_train[:1000])):
    if i % 100 == 0:
        print("Epoch #%d: Loss=%.3f | Accuracy=%.2f%%" % (i, l/100, acc))
        l = 0
        acc = 0
    loss, accuracy = train(img, label, learning_rate=0.005)
    l += loss
    acc += accuracy