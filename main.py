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

def build_model(image):
    # initialize the architecture
    model = convolution.conv2d(image/255)
    model = maxpool.pool(model)
    model = softmax.dense(model)
    
    return model

def compile_model(img, label, learning_rate):
    
    # initialize model
    train_model = build_model(img)

    # forward propagation step
    output, loss, acc = softmax.forward_propagation(img, label, train_model)
    
    # initial gradient
    grad = np.zeros(10) # 10 different classes
    grad[label] = -1 / output[label]
    
    # back propagation
    grad = softmax.back_propagation(grad, learning_rate)
    grad = maxpool.back_propagation(grad)
    grad = convolution.back_propagation(grad, learning_rate)
    
    return loss, acc


def train(epochs, lr, features, labels):
    loss_dict = []
    acc_dict = []
    NUM_EPOCHS = epochs
    learning_rate = lr

    for epoch in range(NUM_EPOCHS):  
        print("============= EPOCH", epoch+1, "=============")
        total_loss = 0
        accuracy = 0

        for i, (img, label) in enumerate(zip(features, labels)): # let's train first 1000 data for simplicity 
            # build the complete model
            if i % 100 == 0 and i != 0:
                print("Step", i, ": Loss= ", total_loss/100, "| Accuracy=", accuracy, "%")

                total_loss = 0
                accuracy = 0
            loss, acc = compile_model(img, label, learning_rate)
            total_loss += loss
            accuracy += acc
            
        loss_dict.append(total_loss)
        acc_dict.append(accuracy)
    
    return loss_dict, acc_dict

def test(X_test, y_test):
    loss = 0
    accuracy = 0
    for img, label in zip(X_test, y_test):
        test_model = build_model(img)
        _, l, acc = softmax.forward_propagation(img, label, test_model, reg_lambda=1e-3)
        loss += l
        accuracy += acc

    print("Test Loss: ", loss/len(X_test))
    print("Test Accuracy: ", accuracy/len(y_test))
    

l, a = train(epochs=3, lr=0.01, features=x_train[:1000], labels=y_train[:1000])