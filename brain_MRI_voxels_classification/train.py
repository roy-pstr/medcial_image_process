##################### Brain MRI Voxels Classification Project #####################
import os
import random
import cv2
from model import model
import numpy as np
import matplotlib.pyplot as plt

#####################  Dataset Creation #####################
def create_dataset(train_or_val):
    # create lables (0 - neg, 1 - pos)
    data_classes = ["neg", "pos"]
    # load images
    data_arr = []
    datadir = os.path.dirname(os.path.abspath(__file__))
    if train_or_val == "training":
        path = os.path.join(datadir, "training")
    else:
        path = os.path.join(datadir, "validation")
    for data_class in data_classes:
        label_num = data_classes.index(data_class)
        for img in os.listdir(path):
            if data_class in img:
                img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # normalize images
                img = img/255
                data_arr.append([img, label_num])
    random.shuffle(data_arr)
    x = []
    y = []
    for img, label in data_arr:
        x.append(img.reshape(-1))
        y.append(label)
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(y.shape[0],1)
    # Normalize the data: subtract the mean image
    mean_image = np.mean(x, axis=0)
    x -= mean_image

    return x, y

#####################  Split to Batches #####################
def split_to_mini_batches(x,y, batch_size=32):
    # split [num_of_inputs, N] -> [num_of_inputs / D, D, N]
    num_of_inputs = x.shape[0]
    N = x.shape[1]
    D = batch_size
    return x.reshape(int(num_of_inputs / D), D, N), y.reshape(int(num_of_inputs / D), D, 1)

def get_random_batch(x,y, batch_size=32):
    rand_indices = np.random.choice(x.shape[0], batch_size)
    return x[rand_indices], y[rand_indices]


#####################  Main  #####################
if __name__ == '__main__':
    np.random.seed(1)

    #     x_train: (512, 1024) y_train: (512, )
    x_train, y_train = create_dataset("training")
    x_val, y_val = create_dataset("validation")
    print("training data: x_train {}, y_train {}".format(x_train.shape, y_train.shape))
    print("validation data: x_val {}, y_val {}".format(x_val.shape, y_val.shape))


    num_iter = 50000
    image_vector_size = 1024
    hidden_layer_size = 1
    output_size = 1
    batch_size = 32
    learning_rate = 1e-2
    reg = 0.001
    std = 1
    verbose = True

    net = model(image_vector_size, batch_size, hidden_layer_size, output_size, std=std)
    stats = net.train(x=x_train, y=y_train, x_val=x_val, y_val=y_val, learning_rate=learning_rate,
                        reg=reg, num_iters=num_iter,
                        batch_size=batch_size, verbose=verbose)

    if verbose:
        # Plot the loss function and train / validation accuracies
        plt.subplot(2, 1, 1)
        plt.plot(stats['loss_history'])
        plt.title('Loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.subplot(2, 1, 2)
        plt.plot(stats['train_acc_history'], label='train')
        plt.plot(stats['val_acc_history'], label='val')
        plt.title('Classification accuracy history')
        plt.xlabel('Epoch')
        plt.ylabel('Classification accuracy')
        plt.legend()
        plt.show()
    print("finished training")
