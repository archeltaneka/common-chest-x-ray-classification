import numpy as np
# import our custom library
from flowtensor import Convolution2D, MaxPool2D, Softmax
from utils import train, test, show_loss_graph, show_acc_graph
# let's test with mnist datasets
from tensorflow.keras.datasets import mnist, fashion_mnist

# load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# define convolution, pooling, and fully connected(dense) layers
convolution = Convolution2D(filter_shape=(3,3), num_filters=8, padding='same', stride=1, kernel_init='none')
maxpool = MaxPool2D(pool_size=2)
temp = convolution.conv2d(x_train[0]/255) # save img dimensions
temp = maxpool.pool(temp)
tmp_shape = temp.shape
softmax = Softmax(tmp_shape[0]*tmp_shape[1]*tmp_shape[2], 10, activation='softmax')

# train the model
l, a = train(convolution, maxpool, softmax, epochs=5, lr=0.01, features=x_train[:1000], labels=y_train[:1000])
# test the model
preds = test(convolution, maxpool, softmax, x_test[:1000], y_test[:1000])
# shows loss & accuracy graph
show_loss_graph(l)
show_acc_graph(a)
