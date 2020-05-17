##################### Brain MRI Voxels Classification Project #####################
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

##################### Sigmoid Activation #####################
def sigmoid(input):
    return 1/(1 + np.exp(-input))
##################### Derivative of a Sigmoid #####################
def d_sigmoid(out):
    sig = sigmoid(out)
    return sig * (1 - sig)

#####################  ReLU Activation #####################
def relu(input):
    return np.maximum(input, 0)
##################### Derivative of a Relu #####################
def d_relu(d_init, out):
    d = np.array(d_init, copy = True)
    d[out < 0] = 0.
    return d

class model:
    def __init__(self, input_size, batch_size, hidden_layer_size, output_size, std=1e-4):
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_layer_size)
        self.params['b1'] = np.zeros(hidden_layer_size)
        self.params['W2'] = std * np.random.randn(hidden_layer_size, output_size)
        self.params['b2'] = np.zeros((batch_size, output_size))

        self.grads = {}
        self.grads['W1'] = np.zeros((input_size, hidden_layer_size))
        self.grads['b1'] = np.zeros(hidden_layer_size)
        self.grads['W2'] = np.zeros((hidden_layer_size, output_size))
        self.grads['b2'] = np.zeros((batch_size, output_size))

        self.z1 = self.h1 = np.zeros((self.batch_size, self.input_size))
        self.z2 = np.zeros(self.batch_size)

    #####################  Forward Function #####################
    def forward(self, x):
        # X: (NxD), N - batch size, D - vectorized image (1024)
        # W1: (DxH), H - hidden layer size, b1: (Hx1)
        # h1: (NxH)
        # W2: (Hx1), H - hidden layer size, b2: (Nx1)
        # y_pred: (Nx1)

        # hidden layer:
        self.z1 = x.dot(self.params['W1']) + self.params['b1']  # fully connected -> (N, H)
        self.h1 = relu(self.z1)  # ReLU

        # output layer:
        self.z2 = self.h1.dot(self.params['W2']) + self.params['b2']  # fully connected - > (N, )
        y_pred = sigmoid(self.z2) # h2

        return y_pred

    #####################  Loss Calculation #####################
    def calc_loss(self, y_pred, y):
        N = y.shape[0]
        # loss:
        loss = (1/2) * np.sum(np.power((y_pred-y), 2))
        # accuracy:
        y_pred = np.round(y_pred)
        accuracy = (y_pred==y)
        return loss, accuracy

    #####################  Backward Function #####################
    def backward(self, x, y_train, y_pred, reg = 5e-6):
        dL_dy = np.array(y_train).reshape(1,1) - np.array(y_pred).reshape(1,1) # [Nx1] ?? -> MSE
        dy_dz2 = d_sigmoid(y_train).reshape(1,1) # [Nx1]
        dL_dz2 = dL_dy * dy_dz2 # [Nx1]

        # W2 gradient (dL_dW2 = dL_dz2 * dz2_dW2)
        dz2_dW2 = self.h1 # [NxH]
        dL_dW2 = np.dot(dz2_dW2.transpose(), dL_dz2) # [HxN] * [Nx1] = [Hx1]

        # b2 gradient (dL_db2 = dL_dz2 * dz2_db2 = dL_dz2)
        dL_db2 = dL_dz2 # [Nx1]

        # W1 gradient (dL_dW1 = dL_dz1 * dz1_dW1)
        dL_dh1 = np.dot(dL_dz2, self.params['W2'].transpose()) # W2 = dz2_dh1, [Nx1] * [1xH] = [NxH]
        #dL_dz1 = dL_dh1*dh1_dz1:
        dL_dz1 = d_relu(dL_dh1, self.z1) # [NxH]
        dL_dW1 = np.dot(x.transpose(), dL_dz1) # [DxN] * [NxH] = [DxH]

        # b1 gradient (dL_db1 = dL_dz1 * dz1_db1 = dL_dz1)
        dL_db1 = dL_dz1.sum(axis=0)  # [Hx1]

        # regularization gradient
        dL_dW1 += reg * 2 * self.params['W1']
        dL_dW2 += reg * 2 * self.params['W2']

        self.grads = {'W1': dL_dW1, 'b1': dL_db1, 'W2': dL_dW2, 'b2': dL_db2}

    #####################  Parameters Update #####################
    def update_parameters(self, lr=1e-3):
        self.params['W1'] -= lr * self.grads['W1']
        self.params['b1'] -= lr * self.grads['b1']
        self.params['W2'] -= lr * self.grads['W2']
        self.params['b2'] -= lr * self.grads['b2']
