import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main

NITER = 10000
DECAY_RATE_LINEAR = .99
DECAY_RATE_CROSS_ENTROPY = .999
DECAY_RATE_MULTI = .8
DECAY_RATE_CNN = .99
LEARNING_RATE_LINEAR = .01
LEARNING_RATE_CROSS_ENTROPY = .1
LEARNING_RATE_MULTI = 1e-2
LEARNING_RATE_CNN = 1e-2

def get_mini_batch(im_train, label_train, batch_size):
    _, train_length = im_train.shape
    num_batches = (train_length // batch_size)
    if train_length % batch_size != 0:
        num_batches += 1

    mini_batch_x = np.zeros(shape=(num_batches, batch_size, 196))
    mini_batch_y = np.zeros(shape=(num_batches, batch_size, 10))

    shuffler = np.random.permutation(train_length)
    shuffled_im_train = im_train[:, shuffler]
    shuffled_label_train = label_train[:, shuffler]
    for i in range(num_batches):
        mini_batch_x[i, :, :] = shuffled_im_train[:, i*32:(i+1)*32].transpose()
        label_lst = shuffled_label_train[0, i*32:(i+1)*32]
        for j, label in enumerate(label_lst):
            encoding = np.zeros(10)
            encoding[label] = 1
            mini_batch_y[i, j, :] = encoding

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    tmp_x = x
    if x.ndim > 1:
        tmp_x = x.flatten()
    y = np.matmul(w, tmp_x) + b.flatten()

    return y


def fc_backward(dl_dy, x, w, b, y):
    dl_dw = np.outer(dl_dy, x.T)
    dl_dx = np.dot(w.T, dl_dy)
    dl_db = np.sum(dl_dy)
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l = np.sum((y_tilde - y) ** 2)
    dl_dy = 2 * (y_tilde - y)
    return l, dl_dy


def loss_cross_entropy_softmax(x, y):
    x_max = x - np.max(x)
    y_tilde = np.exp(x_max) / np.sum(np.exp(x_max))
    l = -1 * np.sum(y * np.log(y_tilde))
    dl_dy = y_tilde - y
    return l, dl_dy


def relu(x):
    y = np.maximum(x, np.zeros(shape=x.shape))
    return y


def relu_backward(dl_dy, x, y):
    dl_dx = dl_dy
    dl_dx[y < 0] = 0
    return dl_dx

## im2col/col2im process found here: https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
## This resource was distributed by TAs by official Canvas announcements to the class
def im2col(x, size):
    ## Assume x is already padded
    ## size = w.shape
    x_j, x_i, x_c = x.shape
    filter_j, filter_i, input_channels, filter_channels = size
    new_x, new_y = (x_j - filter_j) + 1, (x_i - filter_i) + 1
    output = np.zeros(shape=(new_x*new_y, filter_i*filter_j))
    for i in range(new_y):
        for j in range(new_x):
            window = x[i:i+filter_i, j:j+filter_j]
            output[i*new_x+j, :] = window.flatten()
    return output

def col2im(x, output_size):
    y = x.reshape(output_size)
    return y

def conv(x, w_conv, b_conv):
    filter_h, filter_w, _, filter_c = w_conv.shape
    x_reshape = x.reshape(14,14, 1)
    pad_y, pad_x = filter_h // 2, filter_w // 2
    x_pad = np.pad(x_reshape, ((pad_y, pad_y), (pad_x, pad_x), (0,0)), constant_values=0)
    y = np.zeros((x.shape[0],x.shape[1],filter_c))
    x_im2col = im2col(x_pad, w_conv.shape)
    # for i in range(0, y.shape[0], 1):
    #     for j in range(0, y.shape[1], 1):
    #         for k in range(filter_c):
    #             y[i, j, k] = np.sum(w_conv[k, :] * x_pad[i:i+filter_h,j:j+filter_w])
    #             y[i, j, k] += b_conv[k]
    for k in range(filter_c):
        y_im2col = x_im2col.dot(w_conv[:, :, :, k].flatten())
        y_im2col += np.ones(y_im2col.shape) * b_conv.flatten()[k]
        y[:, :, k] = col2im(y_im2col, (x.shape[0], x.shape[1]))
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):

    filter_i, filter_j, _, filter_c = w_conv.shape
    pad_x, pad_y = filter_j // 2, filter_i // 2
    x_reshape = x.reshape((14,14, 1))
    x_pad = np.pad(x_reshape, ((pad_x, pad_x), (pad_y, pad_y),(0,0)), constant_values=0)
    x_im2col = im2col(x_pad, w_conv.shape)
    dy_im2col = dl_dy.reshape((dl_dy.shape[0] * dl_dy.shape[1], dl_dy.shape[2]))
    x_dy = np.zeros((dl_dy.shape[0] * dl_dy.shape[1], filter_j*filter_i, dl_dy.shape[2]))

    for k in range(dl_dy.shape[2]):
        cur_dy = dy_im2col[:,k]
        mul_res = (x_im2col.T * cur_dy).T
        x_dy[:, :, k] = mul_res
    dl_dw = np.sum(x_dy, axis=0)
    dl_dw = dl_dw.reshape((3,3,1,3))

    dl_db = np.zeros(dl_dy.shape[2])
    for k in range(dl_dy.shape[2]):
        dl_db[k] = np.sum(dl_dy[:, :, k])
    return dl_dw, dl_db


def pool2x2(x):
    y = np.zeros(shape=(x.shape[0]//2, x.shape[1]//2, x.shape[2]))
    for k in range(x.shape[2]):
        for i in range(0, x.shape[0], 2):
            for j in range(0, x.shape[1], 2):
                y[i//2, j//2, k] = np.max(x[i:i+2, j:j+2, k])
    return y


def pool2x2_backward(dl_dy, x, y):
    dl_dx = np.zeros(shape=x.shape)
    for k in range(y.shape[2]):
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                ## find index of max in x
                ind = np.argmax(x[2*i:2*i+2, 2*j:2*j+2, k])
                ycor, xcor, kcor = np.unravel_index(ind, x.shape)
                ## set val in dl_dx to corresponding loss value in dl_dy
                dl_dx[ycor+2*i, xcor+2*j, k] = dl_dy[i, j, k]
    return dl_dx


def flattening(x):
    y = x.flatten()
    return y


def flattening_backward(dl_dy, x, y):
    dl_dx = dl_dy.reshape(x.shape)
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    learning_rate = LEARNING_RATE_LINEAR
    decay_rate = DECAY_RATE_LINEAR
    weights = np.random.normal(loc=.5, size=(10,196))
    bias = np.zeros(10).transpose()
    num_batches, batch_size, _ = mini_batch_x.shape
    k = 0
    nIters = NITER
    for iIter in range(nIters):
        if iIter % 1000 == 0:
            learning_rate = decay_rate*learning_rate

        dL_dw, dL_db = np.zeros(shape=weights.shape), np.zeros(shape=bias.shape)
        for ind in range(batch_size):
            im = mini_batch_x[k, ind, :]
            label = mini_batch_y[k, ind, :]
            prediction = fc(im, weights, bias)
            l, dl_dy = loss_euclidean(prediction, label)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, im, weights, bias, prediction)
            dL_dw += dl_dw
            dL_db += dl_db

        k += 1
        if k >= num_batches:
            k = 0
        weights -= learning_rate * dL_dw / batch_size
        bias -= learning_rate * dL_db / batch_size


    return weights, bias


def train_slp(mini_batch_x, mini_batch_y):
    learning_rate = LEARNING_RATE_CROSS_ENTROPY
    decay_rate = DECAY_RATE_CROSS_ENTROPY
    weights = np.random.normal(loc=.5, size=(10,196))
    bias = np.zeros(10).transpose()
    num_batches, batch_size, _ = mini_batch_x.shape
    k = 0
    nIters = NITER
    for iIter in range(nIters):
        if iIter % 1000 == 0:
            learning_rate = decay_rate*learning_rate
        dL_dw, dL_db = np.zeros(shape=weights.shape), np.zeros(shape=bias.shape)
        for ind in range(batch_size):
            im = mini_batch_x[k, ind, :]
            label = mini_batch_y[k, ind, :]
            prediction = fc(im, weights, bias)
            l, dl_dy = loss_cross_entropy_softmax(prediction, label)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, im, weights, bias, prediction)
            dL_dw += dl_dw
            dL_db += dl_db

        k += 1
        if k >= num_batches:
            k = 0
        weights -= learning_rate * dL_dw / batch_size
        bias -= learning_rate * dL_db / batch_size


    return weights, bias


def train_mlp(mini_batch_x, mini_batch_y, learning_rate=LEARNING_RATE_MULTI, decay_rate=DECAY_RATE_MULTI):
    w1 = np.random.normal(loc=.5, size=(30,196))
    b1 = np.zeros(30).transpose()
    w2 = np.random.normal(loc=.5, size=(10,30))
    b2 = np.zeros(10).transpose()
    num_batches, batch_size, _ = mini_batch_x.shape
    k = 0
    nIters = NITER
    iIter = 0
    loss = []
    for iIter in range(nIters):
        if iIter % 1000 == 0 and iIter != 0:
            learning_rate = decay_rate*learning_rate
        if iIter % 4999 == 0 and iIter != 0:
            plt.plot(range(iIter), loss)
            plt.grid()
            plt.show()
        dL_dw_1, dL_db_1, dL_dw_2, dL_db_2 = np.zeros(shape=w1.shape), np.zeros(shape=b1.shape), np.zeros(shape=w2.shape), np.zeros(shape=b2.shape)
        l_sum = np.zeros(32)
        print(iIter,'/', nIters)
        for ind in range(batch_size):
            im = mini_batch_x[k, ind, :]
            label = mini_batch_y[k, ind, :]
            prediction_1 = fc(im, w1, b1)
            relu_res = relu(prediction_1)
            prediction_2 = fc(relu_res, w2, b2)


            l, dl_dx = loss_cross_entropy_softmax(prediction_2, label)
            l_sum[ind] = l
            dl_dx, dl_dw_2, dl_db_2 = fc_backward(dl_dx, relu_res, w2, b2, prediction_2)
            dl_dx = relu_backward(dl_dx, prediction_1, relu_res)
            dl_dx, dl_dw_1, dl_db_1 = fc_backward(dl_dx, im, w1, b1, prediction_1)

            dL_dw_2 += dl_dw_2
            dL_dw_1 += dl_dw_1

            dL_db_1 += dl_db_1
            dL_db_2 += dl_db_2
        k += 1
        if k >= num_batches:
            k = 0
        w1 -= learning_rate * dL_dw_1 / batch_size
        b1 -= learning_rate * dL_db_1 / batch_size

        w2 -= learning_rate * dL_dw_2 / batch_size
        b2 -= learning_rate * dL_db_2 / batch_size
        loss += [np.sum(l_sum)/batch_size]
        print(loss[-1])
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    learning_rate = LEARNING_RATE_CNN
    decay_rate = DECAY_RATE_CNN
    w_conv = np.random.normal(loc=.5, size=(3,3,1,3))
    b_conv = np.random.normal(loc=.5, size=(3))
    w_fc = np.random.normal(loc=.5, size=(10,147))
    b_fc = np.random.normal(loc=.5, size=(10,1))
    num_batches, batch_size, _ = mini_batch_x.shape
    k = 0
    loss = []
    nIters = NITER
    for iIter in range(nIters):
        if iIter % 1000 == 0 and iIter != 0:
            learning_rate = decay_rate * learning_rate
        if iIter % 1000 == 0 and iIter != 0:
            plt.plot(range(iIter), loss)
            plt.grid()
            plt.show()
        dL_dw_conv, dL_db_conv, dL_dw_fc, dL_db_fc = np.zeros(shape=(3,3, 1, 3)), np.zeros(shape=(3)), np.zeros(shape=(10,147)), np.zeros(shape=(10,1))
        print(iIter, '/', nIters)
        l_sum = np.zeros(32)
        for ind in range(batch_size):
            im = mini_batch_x[k, ind, :].reshape((14,14,1))
            label = mini_batch_y[k, ind, :]
            x_conv = conv(im, w_conv, b_conv)
            relu_res = relu(x_conv)
            pool2x2_res = pool2x2(relu_res)
            flatten_res = flattening(pool2x2_res)
            prediction = fc(flatten_res, w_fc, b_fc)
            l, dl_dy = loss_cross_entropy_softmax(prediction, label)
            dl_dy = dl_dy.flatten()
            l_sum[ind] = l
            dl_dx, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, flatten_res, w_fc, b_fc, prediction)
            dl_dx = flattening_backward(dl_dx, pool2x2_res, flatten_res)
            dl_dx = pool2x2_backward(dl_dx, relu_res, pool2x2_res)
            dl_dx = relu_backward(dl_dx, x_conv, relu_res)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dx, im, w_conv, b_conv, x_conv)

            dL_dw_conv += dl_dw_conv
            dL_db_conv += dl_db_conv
            dL_dw_fc += dl_dw_fc
            dL_db_fc += dl_db_fc

        k += 1
        if k >= num_batches:
            k = 0

        w_conv -= learning_rate * dL_dw_conv / batch_size
        b_conv -= learning_rate * dL_db_conv / batch_size

        w_fc -= learning_rate * dL_dw_fc / batch_size
        b_fc -= learning_rate * dL_db_fc / batch_size
        loss += [np.sum(l_sum)/batch_size]
        print(loss[-1])

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    # main.main_slp_linear()
    # main.main_slp()
    # main.main_mlp()
    main.main_cnn()



