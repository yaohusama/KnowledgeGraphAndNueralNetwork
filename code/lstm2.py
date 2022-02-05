# -*- coding: UTF-8 -*-
print(__doc__)


import sys
# import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

from sklearn.neural_network import MLPRegressor

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
import pickle

num = pickle.load(open("Source/count_all.pkl", "rb"))
f = open("Source/data.pkl", "rb")
df = pickle.load(f)
data = df["data"]
print("data.shape:", np.array(data).shape)
label = df['label']
x = list(sorted(set(label)))
# print(x)
label = list(map(lambda xx: x.index(xx), label))
print("max", max(label))
print("label", label)
num_class = len(list(set(label)))
rnn_unit = 200  # hidden layer units
input_size = 400
num_step = 128
output_size = num_step
lr = 0.0006  # 学习率


def get_train_data(batch_size=num_step, time_step=num_step, train_begin=0, train_end=int(num * 0.7)):
    batch_index = []
    data_train = data[train_begin:train_end]
    print("data_train.shape:", np.array(data_train).shape)
    label_train = label[train_begin:train_end]
    # normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    normalized_train_data = data_train
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
            # x = normalized_train_data[i:i + time_step, :7]

            x = data_train[i:i + time_step]
            # y = normalized_train_data[i:i + time_step, 7, np.newaxis]
            y = label_train[i:i + time_step]
            train_x.append(x)
            train_y.append(y)
    # batch_index.append((len(normalized_train_data) - time_step))
    train_x.append(data_train[len(data_train) - time_step:len(data_train)])
    train_y.append(label_train[len(data_train) - time_step:len(data_train)])

    print(np.array(train_x).shape)
    print(np.array(train_y).shape)
    return batch_index, np.array(train_x), np.array(train_y)


# 测试集化分
def get_test_data(time_step=num_step, test_begin=int(num * 0.7)):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    # normalized_test_data = (data_test - mean) / std  # 标准化
    normalized_test_data = data_test
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step]
        y = normalized_test_data[i * time_step:(i + 1) * time_step]
        test_x.append(x)
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:]))
    test_y.extend((normalized_test_data[(i + 1) * time_step:]))
    return mean, std, test_x, test_y



scorelist = []

import keras
# from sklearn.datasets import fetch_mldata
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import LSTM, Dense

nbatch_size = 128
nEpoches = 10
timesteps = 128


def buildrnn():
    pass


def buildlstm():
    import numpy as np

    data_dim = 400
    timesteps = 128
    num_classes = (int)(num_class)

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(128 * num_class, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())
    return model
    pass


def makedata_train():
    img_rows, img_cols = 28, 28

    batch_index, train_x, train_y = get_train_data(nbatch_size, timesteps, 0, (int)(num))

    print("train_y", train_y.shape)
    train_y = np_utils.to_categorical(train_y, num_classes=num_class)
    train_y = train_y.reshape(train_y.shape[0], -1)
    # input_shape = (img_rows, img_cols, 1)
    # x_train, x_test = X[:60000], X[60000:]
    # y_train, y_test = y[:60000], y[60000:]

    return train_x, train_y
    pass




from keras.models import load_model
import os


def runTrain(model, x_train, x_test, y_train, y_test):
    if os.path.exists("model.h5"):
        load_model("model.h5")
    model.fit(x_train, y_train, batch_size=nbatch_size, epochs=nEpoches)
    model.save("model.h5")
    score = model.evaluate(x_test, y_test, batch_size=nbatch_size)
    print('evaluate score:', score)
    pass


def testAccRate(model, x_test, y_test):
    pass


def test():
    X, y = makedata_train()
    tmp = (int)(num * 0.7)
    x_train, x_test = X[:tmp], X[tmp:]
    y_train, y_test = y[:tmp], y[tmp:]
    model = buildlstm()
    runTrain(model, x_train, x_test, y_train, y_test)
    pass




if __name__ == "__main__":
    sys_code_type = sys.getfilesystemencoding()
    # print mystr.decode('utf-8').encode(sys_code_type)
    test()

