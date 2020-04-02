# Common Chest X-ray Classification and Localization

This repository is intended for my final dissertation at the University of Nottingham


## Prerequisites

Download and install the latest [Anaconda]([https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)) according to your OS

Clone this repository, then create a virtual environment using the environment.yaml file provided
```shell
conda env create -f environment.yaml
```
To use the environment, use conda activate (on Windows) or source activate (on OSX/Linux)
```shell
conda activate dissertation_env
```

## Flowtensor
To see the library in action, open your command line, change the directory path to the folder you just cloned, then run
```shell
python main.py
```
======================================================
Creating the model is as easy as
```
convolution = Convolution2D(filter_shape=(3,3), num_filters=8, padding='same', activation='relu', stride=1, kernel_init='random', debugging=False)
maxpool = MaxPool2D(pool_size=2, debugging=False)
softmax = Softmax(13*13*8, 10, activation='softmax', regularizer='l2', debugging=False)
```

Now, let's try your first flowtensor program! :)

First, open up Jupyter Notebook
```shell
jupyter notebook
```
the command above will open the Jupyter Notebook in a new tab in your browser. Then, create a new notebook (.ipynb) file with Python 3. Make sure you create the new file in the same directory with the repository you just cloned.
<br>
We are going to use a simple dataset from the MNIST handwritten digits
```
import numpy as np
from lib.flowtensor import Convolution2D, MaxPool2D, Softmax
from lib.utils import train, test, show_loss_graph, show_acc_graph
from tensorflow.keras.datasets import mnist, fashion_mnist

# load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# define convolution, pooling, and fully connected(dense) layers
convolution = Convolution2D(filter_shape=(3,3), num_filters=8, padding='same', activation='relu', stride=1, kernel_init='random', debugging=False)
maxpool = MaxPool2D(pool_size=2, debugging=False)
temp = convolution.conv2d(x_train[0]/255) # save img dimensions
temp = maxpool.pool(temp)
tmp_shape = temp.shape
softmax = Softmax(tmp_shape[0]*tmp_shape[1]*tmp_shape[2], 10, activation='softmax', regularizer='l2', debugging=False)

# train the model
l, a = train(convolution, maxpool, softmax, epochs=5, lr=0.01, features=x_train[:1000], labels=y_train[:1000])

# test the model
preds = test(convolution, maxpool, softmax, x_test[:1000], y_test[:1000])

# shows loss & accuracy graph
show_loss_graph(l)
show_acc_graph(a)
```

## Model Training

Under 'Kaggle' directory, you can find 2 separate files. I recommend you to run these files on Kaggle ([NIH dataset]([https://www.kaggle.com/nih-chest-xrays/data](https://www.kaggle.com/nih-chest-xrays/data)) and [Pneumonia dataset]([https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia))). Register or sign in if you already have an account. All dataset and library has been set up for you. Create new notebook and just upload the notebook file from this repository and you are good to go.
<div align="center">
  <img src="../img/kaggle_new_notebook.PNG">
</div>
The most important thing in Kaggle is access to GPU computations. Just be aware of your weekly quota. Each account is only given 30 hours per week.

## Predicting and Visualizing Chest X-ray Diseases

Under the 'app' directory, there are 2 separate notebook files. Before that, make sure you already have the weights that you saved while training the model. Use the weights you trained from NIH dataset on the multiprediction and Pneumonia dataset on uniprediction. I could not provide the weights because Github only allows 100MB max file size upload.
You should probably see the output like this
<div align="center">
  <img src="../img/output.PNG">
</div>