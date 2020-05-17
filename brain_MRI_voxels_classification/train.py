##################### Brain MRI Voxels Classification Project #####################
import os
import random
import cv2
from model import model
import numpy as np

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
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # normalize images
                img_array = img_array/255
                data_arr.append([img_array, label_num])
    random.shuffle(data_arr)
    x = []
    y = []
    for features, label in data_arr:
        x.append(features.reshape(-1))
        y.append(label)
    return np.array(x), np.array(y)

#####################  Split to Batches #####################
def split_to_batches(x,y, batch_size=32):
    # split [num_of_inputs, N] -> [num_of_inputs / D, D, N]
    num_of_inputs = x.shape[0]
    N = x.shape[1]
    D = batch_size
    return x.reshape(int(num_of_inputs / D), D, N), y.reshape(int(num_of_inputs / D), D)

#####################  Main  #####################
if __name__ == '__main__':
    num_epochs = 100
#     x_train: (512, 1024) y_train: (512, )
    x_train, y_train = create_dataset("training")
    x_val, y_val = create_dataset("validation")
    print(type(x_train))
    print("training data: x_train {}, y_train {}".format(x_train.shape, y_train.shape))
    print("validation data: x_val {}, y_val {}".format(x_val.shape, y_val.shape))

    verbose = True
    learning_rate = 1e-3
    batch_size = 1
    our_model = model(1024, batch_size, 10, 1)

    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        x_train_batches, y_train_batches = split_to_batches(x_train, y_train, batch_size)
        accuracy_per_epoch = []
        for i in range(x_train_batches.shape[0]):
            x_mini_batch= x_train_batches[i]
            y_mini_batch=y_train_batches[i]
            y_pred = our_model.forward(x_mini_batch)
            loss, accuracy = our_model.calc_loss(y_pred, y_mini_batch)
            loss_history.append(loss)
            accuracy_per_epoch.extend(accuracy)
            our_model.backward(x_mini_batch, y_mini_batch, y_pred)
            our_model.update_parameters(learning_rate)

        ################ print current status #########################
        if verbose:
            print('epoch %d / %d: loss %f' % (epoch, num_epochs, loss))
            # Every epoch, check train and val accuracy:
            # Check accuracy
            train_acc = (np.round(our_model.forward(x_mini_batch)) == y_mini_batch).mean()
            val_acc = (np.round(our_model.forward(x_val[:32,:])) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            print("loss history: {}".format(loss_history))
            print("train accuracy history: {}".format(train_acc_history))
            print("validate accuracy history: {}".format(val_acc_history))

    print("finished training")
