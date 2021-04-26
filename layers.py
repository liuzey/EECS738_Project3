import os
import numpy as np
import copy


class ConvolutionLayer:
    def __init__(self, kernel=(3, 3), channels=32, stride=(1, 1), padding=(0, 0)):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.channels = channels
        self.weight = np.ones((self.channels, kernel[0], kernel[1])) * 0.01
        self.bias = np.zeros(self.channels)

    def forward(self, inputs):
        self.inputs = inputs
        self.dim = [(inputs.shape[i+2]-self.kernel[i]+2 * self.padding[i])//self.stride[i] + 1 for i in range(2)]
        res = np.zeros((inputs.shape[0], self.channels, self.dim[0], self.dim[1]))
        padded_inputs = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2]+2*self.padding[0], inputs.shape[3]+2*self.padding[1]))
        padded_inputs[:, :, self.padding[0]:padded_inputs.shape[2]-self.padding[0], \
                                        self.padding[1]:padded_inputs.shape[3]-self.padding[1]] = inputs
        self.padded_inputs = padded_inputs

        for index in range(inputs.shape[0]):
            for x in range(self.dim[0]):
                for y in range(self.dim[1]):
                    temp = np.array([np.multiply(self.weight, padded_inputs[index, c, x*self.stride[0]:x*self.stride[0]
                                    +self.kernel[0], y*self.stride[1]:y*self.stride[1]+self.kernel[1]]).reshape(self.channels, -1)
                                                            for c in range(inputs.shape[1])])
                    res[index, :, x, y] = np.sum(np.sum(temp, axis=0), axis=1) + self.bias
        self.res = res
        return res

    def backward(self, err):
        inputs = self.padded_inputs
        next_error = np.zeros_like(inputs)
        for index in range(next_error.shape[0]):
            for x in range(self.dim[0]):
                for y in range(self.dim[1]):
                    for c in range(inputs.shape[1]):
                        next_error[index, c, x*self.stride[0]:x*self.stride[0]+self.kernel[0],
                                y*self.stride[1]:y*self.stride[1]+self.kernel[1]] += self.weight[c] * err[index,c,x,y]
                        self.weight[c] += 1e-3 * np.dot(inputs[index, c, x*self.stride[0]:x*self.stride[0]+self.kernel[0],
                                y*self.stride[1]:y*self.stride[1]+self.kernel[1]].T, err[index,c,x,y]).T
        self.bias += 1e-3 * np.sum(err, axis=(0,2,3))
        next_error = next_error[:, :, self.padding[0]:next_error.shape[2] - self.padding[0],
                                self.padding[1]:next_error.shape[3] - self.padding[1]]
        return next_error


class FullyConnectLayer:
    def __init__(self, channels=(100, 100)):
        self.channels = channels
        self.weight = np.random.randn(self.channels[0], self.channels[1])
        self.bias = np.zeros(self.channels[1])

    def forward(self, inputs):
        self.inputs = inputs
        # res = np.zeros((inputs.shape[0], self.channels[1]))
        res = np.dot(inputs, self.weight) + self.bias
        self.res = res
        return res

    def backward(self, err):
        next_error = np.dot(err, self.weight.T)
        self.weight += 1e-3 * np.dot(self.inputs.T, err)
        self.bias += 1e-3 * np.sum(err.T, axis=1)
        return next_error