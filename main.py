from convolution import Convolution2D
from pooling import MaxPool2D
from fully_connected import Softmax

import numpy as np
from tensorflow.keras.datasets import mnist

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# define the model
convolution = Convolution2D(filter_shape=(3,3), num_filters=8, padding='same', stride=1)
maxpool = MaxPool2D(pool_size=2)
temp = convolution.conv2d(x_train[0]/255)
temp = maxpool.pool(temp)
temp_size = temp.shape
softmax = Softmax(temp_size[0]*temp_size[1]*temp_size[2], 10, activation='softmax')

# train helper function
def train(img, label, learning_rate):
    # initialize model
    model = convolution.conv2d(img/255)
    model = maxpool.pool(model)
    model = softmax.dense(model)
    
    # forward propagation
    output, loss, acc = softmax.forward_propagation(img, label, model, reg_lambda=1e-3)
    
    # initial gradient
    grad = np.zeros(10) # 10 different classes
    grad[label] = -1 / output[label]
    
    # back propagation
    grad = softmax.back_propagation(grad, learning_rate, reg_lambda=1e-3)
    grad = maxpool.back_propagation(grad)
    grad = convolution.back_propagation(grad, learning_rate)
    
    return loss, acc

# train the model
NUM_EPOCHS = 3
learning_rate = 0.001

for epoch in range(NUM_EPOCHS):  
    print("============= EPOCH", epoch+1, "=============")
    total_loss = 0
    accuracy = 0

    for i, (img, label) in enumerate(zip(x_train[:1000], y_train[:1000])): # let's train first 1000 data for simplicity 
        # build the complete model
        if i % 100 == 0 and i != 0:
            print("Step", i, ": Loss= ", total_loss/100, "| Accuracy=", accuracy, "%")

            total_loss = 0
            accuracy = 0
        loss, acc = train(img, label, learning_rate=0.001)
        total_loss += loss
        accuracy += acc
        
# test the model
loss = 0
accuracy = 0
for img, label in zip(x_test[:1000], y_test[:1000]):
    model = convolution.conv2d(img/255)
    model = maxpool.pool(model)
    model = softmax.dense(model)
    _, l, acc = softmax.forward_propagation(img, label, model, reg_lambda=1e-3)
    loss += l
    accuracy += acc
            
print("Test Loss: ", loss/len(x_test[:1000]))
print("Test Accuracy: ", accuracy/len(x_test[:1000]))