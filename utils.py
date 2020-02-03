import numpy as np
import matplotlib.pyplot as plt

def padding(img, pad):
    img_pad = np.pad(img, ((pad, pad), (pad,pad)), 'constant', constant_values=0)
    
    return img_pad

def load_data():
    (x1, y1), (x2, y2) = mnist.load_data()
    return (x1, y1), (x2, y2)

def build_model(c, m, s, img):
    train_model = c.conv2d(img/255)
    train_model = m.pool(train_model)
    train_model = s.dense(train_model)
    
    return train_model

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


def train(c, m, s, epochs, lr, features, labels):
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
                print("Step", i, ": Loss=", total_loss/100, "| Accuracy=", accuracy, "%")

                total_loss = 0
                accuracy = 0
            loss, acc = compile_model(c, m, s, img, label, learning_rate)
            total_loss += loss
            accuracy += acc
            
        loss_dict.append(total_loss)
        acc_dict.append(accuracy)
    
    return loss_dict, acc_dict

def test(c, m, s, X_test, y_test):
    loss = 0
    accuracy = 0
    for img, label in zip(X_test, y_test):
        test_model = build_model(c, m, s, img)
        _, l, acc = s.forward_propagation(img, label, test_model, reg_lambda=1e-3)
        loss += l
        accuracy += acc

    print("Test Loss: ", loss/len(X_test))
    print("Test Accuracy: ", accuracy/len(y_test))
    
def show_loss_graph(l):
    plt.plot([i / 100 for i in l])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
def show_acc_graph(a):
    plt.plot(a)
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()