import os
import numpy as np
import copy
from layers import ConvolutionLayer, FullyConnectLayer
from math import log


class MLP:
    def __init__(self, class_num):
        self.train = True
        self.loss = 0
        self.fc1 = FullyConnectLayer(channels=(28*28, 100))
        self.fc2 = FullyConnectLayer(channels=(100, class_num))
        self.sigmoid1 = Sigmoid()
        self.sigmoid2 = Sigmoid()
        self.dropout1 = DropOut()

    def forward(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        x = x / np.amax(x)
        for bn in [self.dropout1, ]:
            bn.train = self.train

        self.block1 = [self.fc1.forward, self.sigmoid1.forward]
        self.block2 = [self.fc2.forward, self.sigmoid2.forward]

        for block in [self.block1, self.block2]:
            for item in block:
                x = item(x)
                # print(x[0].reshape(-1))
        return x

    def backward(self, err):
        err = self.sigmoid2.backward(err)
        err = self.fc2.backward(err)
        err = self.sigmoid1.backward(err)
        _ = self.fc1.backward(err)

    def save(self, saved_dir):
        np.save(saved_dir + 'fc1.npy', self.fc1.weight)
        np.save(saved_dir + 'fc2.npy', self.fc2.weight)

    def load(self, load_dir):
        self.fc1.weight = np.load(load_dir + 'fc1.npy', allow_pickle=True)
        self.fc2.weight = np.load(load_dir + 'fc2.npy', allow_pickle=True)

    @staticmethod
    def relu(x):
        return np.where(x > 0, x, 0)

    @staticmethod
    def maxpooling(x):
        res = np.zeros((x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]//2))
        for a in range(res.shape[0]):
            for b in range(res.shape[1]):
                for c in range(res.shape[2]):
                    for d in range(res.shape[3]):
                        res[a, b, c, d] = np.max(x[a, b, c*2+2, d*2+2].reshape(-1))
        return res

    @staticmethod
    def loss_(vector, label):
        _loss = np.zeros_like(vector)
        vector_ = np.zeros_like(vector)
        loss_c = np.zeros_like(vector)
        for i in range(vector.shape[0]):
            for j in range(vector.shape[1]):
                _loss[i, j] += -(vector[i, j])
                vector_[i, j] = np.exp(vector[i, j]) / sum(np.exp(vector[i]))
            try:
                _loss[i, int(label[i])] += 1.0
                loss_c[i, int(label[i])] += -log(vector_[i, int(label[i])])
            except ValueError:
                print(vector[i])
                exit()
        return _loss, loss_c


class DropOut:
    def __init__(self):
        self.train = True

    def forward(self, x):
        if self.train:
            self.pos = np.random.binomial(1, 0.5, size=x.shape)
            return x * self.pos + np.zeros_like(x)
        return x * 0.5

    def backward(self, err):
        return err * self.pos


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        self.res = 1.0 / (1.0 + np.exp(-x))
        return self.res

    def backward(self, err):
        return err * (self.res * (1.0 - self.res))




