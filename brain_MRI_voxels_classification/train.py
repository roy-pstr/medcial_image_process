import numpy as np

#sigmoid activation
def sigmoid(input):
    return 1/(1 + np.exp(-input))

#derivate of a sigmoid w.r.t. input
def d_sigmoid(out):
    sig = sigmoid(out)
    return sig * (1 - sig)

#relu activation
def relu(input):
    return np.maximum(input, 0)

def d_relu(d_init, out):
    d = np.array(d_init, copy = True)
    d[out < 0] = 0.
    return d

def create_dataset():
    # load images
    # normalize images
    # create lables (0 - neg, 1 - pos)
    return images, labels

def forawrd(X):
    # X: (NxD), N - batch size, D - vectorized image (1024)
    # W1: (DxH), H - hidden layer size, b1: (Hx1)
    # h1: (NxH)
    # W2: (Hx1), H - hidden layer size, b2: (Nx1)
    # y_pred: (Nx1)

    # hidden layer:
    z1 = X.dot(W1) + b1  # fully connected -> (N, H)
    h1 = relu(z1)  # ReLU

    # output layer:
    z2 = h1.dot(W2) + b2  # fully connected - > (N, )
    y_pred = sigmoid(z2)

    return y_pred

def calc_loss(y_pred, y_train):
    loss = np.sqrt(np.sum(np.power((y_pred-y),2)))
    # accuracy -> ...
    return loss, accuracy

def backward(y_train, y_pred):
    dL_dy = (y_train - y_pred) # [Nx1] ??
    dy_dz2 = d_sigmoid(y_train) # [Nx1]
    dL_dz2 = dL_dy * dy_dz2 # [Nx1]

    # W2 gradient (dL_dW2 = dL_dz2 * dz2_dW2)
    dz2_dW2 = h1 # [NxH]
    dL_dW2 = np.dot(dz2_dW2.transpose(), dL_dz2) # [HxN] * [Nx1] = [Hx1]

    # b2 gradient (dL_db2 = dL_dz2 * dz2_db2 = dL_dz2)
    dL_db2 = dL_dz2 # [Nx1]

    # W1 gradient (dL_dW1 = dL_dz1 * dz1_dW1)
    dL_dh1 = np.dot(W2, dL_dz2.transpose()) # W2 = dz2_dh1, [Hx1] * [1xN] = [NxH]
    #dL_dz1 = dL_dh1*dh1_dz1:
    dL_dz1 = d_relu(dL_dh1, z1) # [NxH]
    dL_dW1 = np.dot(X.transpose(), dL_dz1) # [DxN] * [NxH] = [DxH]

    # b1 gradient (dL_db1 = dL_dz1 * dz1_db1 = dL_dz1)
    dL_db1 = dL_dz1.sum(axis=0)  # [Hx1]

    # regularization gradient
    dW1 += reg * 2 * W1
    dW2 += reg * 2 * W2

    grads = {'W1': dL_dW1, 'b1': dL_db1, 'W2': dL_dW2, 'b2': dL_db2}

    return grads

def update_parameters(grads, lr):
    W1 -= lr * grads['dW1']
    b1 -= lr * grads['db1']
    W2 -= lr * grads['dW2']
    b2 -= lr * grads['db2']

if __name__ == '__main__':
    # x_train: (512, 1024) y_train: (512, )
    [x_train, y_train], [x_val, y_val] = create_dataset()
    for epoch in range(num_epochs):
        for mini_batch in train_batches:
            #input_batch: (batch_size, 1024)
            # scores: (batch_size, )
            y_pred = forward(mini_batch[0])
            loss, accuracy = calc_loss(y_pred, mini_batch[1])
            grads = backward() #dW1, db1, dW2, db2
            update_parameters(grads, lr) #update weights and bais

    save_network()
