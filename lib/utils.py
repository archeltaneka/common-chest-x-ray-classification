import numpy as np
import matplotlib.pyplot as plt
import time

'''
    Image padding
    
    # Arguments
        img: 2D tuple/list which is 2D array of an image
        pad: size of padding
     
    # Input
        (nxn) tuple/list
    
    # Output
        (nxn) padded tuple/list
    
    *See usage on flowtensor.Convolutional2D.conv2d
'''
def padding(img, pad):
    img_pad = np.pad(img, ((pad, pad), (pad,pad)), 'constant', constant_values=0)
    
    return img_pad

'''
    Build a neural network model
    
    # Arguments
        c: flowtensor.Convolution2D object of convolutional model
        m: flowtensor.MaxPool2D object of pooling model
        s: flowtensor.Softmax object of dense model
        img: mxn array/tuple of an image.

'''
def build_model(c, m, s, img):
    train_model = c.conv2d(img/255)
    train_model = m.pool(train_model)
    train_model = s.dense(train_model)
    
    return train_model


'''
    Compile a neural network model
    
    # Arguments
        c: flowtensor.Convolution2D object of convolutional model
        m: flowtensor.MaxPool2D object of pooling model
        s: flowtensor.Softmax object of dense model
        img: mxn array/tuple of an image.
        label: nx1 array/list. Label of the image
        learning_rate: Integer. Specifies how fast/slow the model learns

'''
def compile_model(c, m, s, img, label, learning_rate):
    model = build_model(c, m, s, img)
    # forward propagation step
    output, loss, acc = s.forward_propagation(img, label, model)
    
    # initial gradient
    grad = np.zeros(10) # 10 different classes
    grad[label] = -1 / output[label]
    
    # back propagation
    grad = s.back_propagation(grad, learning_rate)
    grad = m.back_propagation(grad)
    grad = c.back_propagation(grad, learning_rate)
    
    return loss, acc

'''
    Train a model
    
    # Arguments:
        c: flowtensor.Convolution2D object of convolutional model
        m: flowtensor.MaxPool2D object of pooling model
        s: flowtensor.Softmax object of dense model
        epochs: Integer. Specifies the iteration through the entire dataset
        lr: Float. Specifies the learning rate of the model
        features: mxn array/tuple. Training features
        labels: nx1 array/list. Training labels
    
'''
def train(c, m, s, epochs, lr, features, labels):
    loss_dict = []
    acc_dict = []
    NUM_EPOCHS = epochs
    learning_rate = lr

    for epoch in range(NUM_EPOCHS):
        start = time.time() # starts the wall time
        print("============= EPOCH", epoch+1, "=============")
        total_loss = 0
        accuracy = 0

        for i, (img, label) in enumerate(zip(features, labels)):
            # build the complete model
            if i % 100 == 0 and i != 0:
                print("Step", i, ": Loss=", total_loss/100, "| Accuracy=", accuracy, "%")

                total_loss = 0
                accuracy = 0
            loss, acc = compile_model(c, m, s, img, label, learning_rate)
            total_loss += loss
            accuracy += acc
            
        loss_dict.append(total_loss)
        acc_dict.append(accuracy)
        
        end = time.time()
        # display time taken for training
        print("Training time:", int(end-start), "s")
    
    return loss_dict, acc_dict

'''
    Test a model
    
    # Arguments:
        c: flowtensor.Convolution2D object of convolutional model
        m: flowtensor.MaxPool2D object of pooling model
        s: flowtensor.Softmax object of dense model
        X_test: mxn array/tuple. Test data
        y_test: nx1 array/list. Test label
    
'''
def test(c, m, s, X_test, y_test):
    loss = 0
    accuracy = 0
    for img, label in zip(X_test, y_test):
        test_model = build_model(c, m, s, img)
        _, l, acc = s.forward_propagation(img, label, test_model, reg_lambda=1e-2)
        loss += l
        accuracy += acc

    print("Test Loss: ", loss/len(X_test))
    print("Test Accuracy: ", accuracy/len(y_test), "(", accuracy/len(y_test) * 100, "% )")

def predict(c, m, s, img, label):
    test_model = build_model(c, m, s, img)
    out, _, _ = s.forward_propagation(img, label, test_model, reg_lambda=1e-2)
    
    return np.argmax(out)
'''
    Display/plot a list of loss
    
    # Arguments:
        l: array/list of loss
'''
def show_loss_graph(l):
    plt.plot([i / 100 for i in l])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

'''
    Display/plot a list of accuracy
    
    # Arguments:
        l: array/list of accuracy
'''
def show_acc_graph(a):
    plt.plot(a)
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()