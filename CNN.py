import os
import numpy as np
import copy
from layers import ConvolutionLayer, FullyConnectLayer
from math import log


class CNN:
    def __init__(self, class_num):
        self.train = True
        self.loss = 0
        self.conv1 = ConvolutionLayer(kernel=(3, 3), channels=32, stride=(1, 1), padding=(0, 0))
        self.conv2 = ConvolutionLayer(kernel=(3, 3), channels=64, stride=(2, 2), padding=(1, 1))
        self.conv3 = ConvolutionLayer(kernel=(3, 3), channels=64, stride=(1, 1), padding=(0, 0))
        self.conv4 = ConvolutionLayer(kernel=(3, 3), channels=128, stride=(2, 2), padding=(1, 1))
        self.conv5 = ConvolutionLayer(kernel=(3, 3), channels=128, stride=(1, 1), padding=(0, 0))
        self.fc1 = FullyConnectLayer(channels=(3200, 100))
        self.fc2 = FullyConnectLayer(channels=(100, class_num))
        self.bn1 = BatchNormalization2d(32)
        self.bn2 = BatchNormalization2d(64)
        self.bn3 = BatchNormalization2d(128)
        self.dropout1 = DropOut()

    def forward(self, x):
        batch = x.shape[0]
        for bn in [self.bn1, self.bn2, self.bn3, self.dropout1]:
            bn.train = self.train

        self.block1 = [self.conv1.forward, self.bn1.forward, self.relu]
        self.block2 = [self.conv2.forward, self.conv3.forward, self.bn2.forward, self.relu, self.dropout1.forward]
        self.block3 = [self.conv4.forward, self.conv5.forward, self.bn3.forward, self.relu]
        self.block4 = [self.fc1.forward, self.relu, self.fc2.forward]

        for block in [self.block1, self.block2, self.block3]:
            for item in block:
                x = item(x)
                # print(x[0].reshape(-1))
        x = x.reshape(batch, -1)
        for item in self.block4:
            x = item(x)
            # print(x[0].reshape(-1))
        return x

    def backward(self, err):
        err = self.fc2.backward(err)
        err *= np.where(self.fc2.inputs > 0, 1.0, 0.0)
        err = self.fc1.backward(err)
        err *= np.where(self.fc1.inputs > 0, 1.0, 0.0)
        err = err.reshape(128, 128, 5, 5)
        err = self.bn3.backward(err)
        err = self.conv5.backward(err)
        err = self.conv4.backward(err)
        err = self.dropout1.backward(err)
        err *= np.where(self.conv4.inputs > 0, 1.0, 0.0)
        err = self.bn2.backward(err)
        err = self.conv3.backward(err)
        err = self.conv2.backward(err)
        err *= np.where(self.conv2.inputs > 0, 1.0, 0.0)
        err = self.bn1.backward(err)
        _ = self.conv1.backward(err)


    def save(self, saved_dir):
        np.save(saved_dir + 'conv1.npy', self.conv1.weight)
        np.save(saved_dir + 'conv2.npy', self.conv2.weight)
        np.save(saved_dir + 'conv3.npy', self.conv3.weight)
        np.save(saved_dir + 'conv4.npy', self.conv4.weight)
        np.save(saved_dir + 'conv5.npy', self.conv5.weight)
        np.save(saved_dir + 'fc1.npy', self.fc1.weight)
        np.save(saved_dir + 'fc2.npy', self.fc2.weight)
        np.save(saved_dir + 'bn.npy', np.array([self.bn1.gamma, self.bn1.beta, self.bn2.gamma, self.bn2.beta, \
                                                self.bn3.gamma, self.bn3.beta]))

    def load(self, load_dir):
        self.conv1.weight = np.load(load_dir + 'conv1.npy', allow_pickle=True)
        self.conv2.weight = np.load(load_dir + 'conv2.npy', allow_pickle=True)
        self.conv3.weight = np.load(load_dir + 'conv3.npy', allow_pickle=True)
        self.conv4.weight = np.load(load_dir + 'conv4.npy', allow_pickle=True)
        self.conv5.weight = np.load(load_dir + 'conv5.npy', allow_pickle=True)
        self.fc1.weight = np.load(load_dir + 'fc1.npy', allow_pickle=True)
        self.fc2.weight = np.load(load_dir + 'fc2.npy', allow_pickle=True)
        bn_paras = np.load(load_dir + 'bn.npy', allow_pickle=True)
        self.bn1.gamma = bn_paras[0]
        self.bn1.beta = bn_paras[1]
        self.bn2.gamma = bn_paras[2]
        self.bn2.beta = bn_paras[3]
        self.bn3.gamma = bn_paras[4]
        self.bn3.beta = bn_paras[5]

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
                vector_[i, j] = np.exp(vector[i, j]) / sum(np.exp(vector[i]))
                _loss[i, j] += -(vector_[i, j])
            try:
                _loss[i, int(label[i])] += (1.0 + 2 * (vector_[i, int(label[i])]))
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


class BatchNormalization2d:
    def __init__(self, n):
        self.n = n
        self.train = True
        self.gamma = np.ones(self.n)
        self.beta = np.zeros(self.n)
        self.mean = np.zeros(self.n)
        self.var = np.zeros(self.n)

    def forward(self, x):
        y = np.zeros_like(x)
        x_nor = np.zeros_like(x)
        if self.train:
            self.mu = np.mean(x, axis=(0,2,3))
            self.theta = np.var(x, axis=(0,2,3))
            self.std_inv = 1.0 / np.sqrt(self.theta + 1e-12)
            for i in range(self.n):
                x_nor[:, i, :, :] = (x[:, i, :, :] - np.ones_like(x[:, 0, :, :]) * self.mu[i]) * self.std_inv[i]
            self.x_nor = x_nor
            self.mean = 0.9 * self.mean + 0.1 * self.mu
            self.var = 0.9 * self.var + 0.1 * self.theta
            for i in range(self.n):
                y[:, i, :, :] = self.x_nor[:, i, :, :] * self.gamma[i] + np.ones_like(x[:, 0, :, :]) * self.beta[i]
        else:
            std_inv = 1.0 / np.sqrt(self.var + 1e-12)
            for i in range(self.n):
                x_nor[:, i, :, :] = (x[:, i, :, :] - np.ones_like(x[:, 0, :, :]) * self.mu[i]) * std_inv[i]
            for i in range(self.n):
                y[:, i, :, :] = x_nor[:, i, :, :] * self.gamma[i] + np.ones_like(x[:, 0, :, :]) * self.beta[i]
        return y

    def backward(self, dy):
        dx_nor = np.zeros_like(dy)
        for i in range(self.n):
            dx_nor[:, i, :, :] = dy[:, i, :, :] * self.gamma[i]
        dx = np.zeros_like(dx_nor)
        for i in range(self.n):
            dx[:, i, :, :] = self.std_inv[i] * (dx_nor - np.mean(dx_nor, axis=0)
                                                - self.x_nor * np.mean(dx_nor * self.x_nor, axis=0))[:, i, :, :]
        dgamma = np.sum(dy * self.x_nor, axis=(0, 2, 3))
        dbeta = np.sum(dy, axis=(0, 2, 3))
        self.gamma += dgamma
        self.beta += dbeta
        return dx


class BatchNormalization1d:
    def __init__(self, n):
        self.n = n
        self.train = True
        self.gamma = np.ones(self.n)
        self.beta = np.zeros(self.n)
        self.mean = np.zeros(self.n)
        self.var = np.zeros(self.n)

    def forward(self, x):
        y = np.zeros_like(x)
        x_1d = copy.deepcopy(x).reshape(x.shape[0], -1)
        if self.train:
            self.mu = np.mean(x_1d, axis=0)
            self.theta = np.var(x_1d, axis=0)
            self.std_inv = 1.0 / np.sqrt(self.theta + 1e-12)
            self.x_nor = (x - self.mu) * self.std_inv
            self.mean = 0.9 * self.mean + 0.1 * self.mu
            self.var = 0.9 * self.var + 0.1 * self.theta
            for i in range(self.n):
                y[:, i] = self.x_nor[:, i] * self.gamma[i] + np.ones_like(x[:, 0]) * self.beta[i]
        else:
            std_inv = 1.0 / np.sqrt(self.var + 1e-12)
            x_nor = (x - self.mean) * std_inv
            for i in range(self.n):
                y[:, i] = x_nor[:, i] * self.gamma[i] + np.ones_like(x[:, 0]) * self.beta[i]
        return y

    def backward(self, dy):
        dx_nor = dy * self.gamma
        dx = self.std_inv * (dx_nor - np.mean(dx_nor, axis=0) - self.x_nor * np.mean(dx_nor * self.x_nor, axis=0))
        dgamma = np.sum(dy * self.x_nor, axis=0)
        dbeta = np.sum(dy, axis=0)
        self.gamma += dgamma
        self.beta += dbeta
        return dx




