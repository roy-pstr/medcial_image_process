import numpy as np

def create_dataset():
    # load images
    # normalize images
    # create lables (0 - neg, 1 - pos)
    return images, labels

def forawrd(X):
    # X: (N, D), N - batch size, D - vectorized image (1024)
    # W1: (D, H), H - hidden layer size, b1: (H, )
    # X2: (N, H)
    # W2: (H, ), H - hidden layer size, b2: (N, )
    # scores: (N, )

    # hidden layer:
    fc1 = X.dot(W1) + b1  # fully connected -> (N, H)
    X2 = np.maximum(0, fc1)  # ReLU

    # output layer:
    scores = X2.dot(W2) + b2  # fully connected - > (N, )
    scores = sigmoid(scores)

    return scores

def calc_loss(y_pred, y):
    loss = np.sqrt(np.sum(np.power((y_pred-y),2)))
    # accuracy -> ...
    return loss, accuracy

def backward(y):
    softmax[np.arange(N), y] -= 1
    softmax /= N

    # W2 gradient
    dW2 = X2.transpose().dot(softmax)  # [HxN] * [NxC] = [HxC]

    # b2 gradient
    db2 = softmax.sum(axis=0)

    # W1 gradient
    dW1 = softmax.dot(W2.transpose())  # [NxC] * [CxH] = [NxH]
    dfc1 = dW1 * (fc1 > 0)  # [NxH] . [NxH] = [NxH]
    dW1 = X.transpose().dot(dfc1)  # [DxN] * [NxH] = [DxH]

    # b1 gradient
    db1 = dfc1.sum(axis=0)

    # regularization gradient
    dW1 += reg * 2 * W1
    dW2 += reg * 2 * W2

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

def step():
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

if __name__ == '__main__':
    # x_train: (512, 1024) y_train: (512, )
    [x_train, y_train], [x_val, y_val] = create_dataset()
    for iter in NUM_OF_ITERS:
        #input_batch: (batch_size, 1024)
        [x_train_batch, y_train_batch] = random_batch(dataset)
        # scores: (batch_size, )
        y_pred = forward(x_train_batch)
        loss, accuracy = calc_loss(y_pred, y_train_batch)
        grads = backward() #dW1, db1, dW2, db2
        step() #update weights and bais

    save_network()
