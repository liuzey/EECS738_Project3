import os
import numpy as np
import copy
import pandas as pd
from PIL import Image
import pickle
import random

BATCH_SIZE = 128


class GDataLoader:
    def __init__(self, data_dir, train, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        if train:
            self.sub_dir, self.csv_name = 'trainingset', 'training.csv'
        else:
            self.sub_dir, self.csv_name = 'testset', 'test.csv'
        csv_path = os.path.join(data_dir, self.sub_dir, self.csv_name)
        self.csv_data = pd.read_csv(csv_path)
        self.data_num = len(self.csv_data)

    def get_item(self, idx):
        img_path = os.path.join(self.data_dir, self.sub_dir, self.csv_data.iloc[idx, 0])
        _img = Image.open(img_path)
        _class = self.csv_data.iloc[idx, 1]

        _img = _img.resize((32, 32))
        _img = np.array(_img)/255
        _img = self.normalize(_img)
        return _img, _class

    def normalize(self, x):
        # (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)
        reshape_x = np.zeros((x.shape[2], x.shape[0], x.shape[1]))
        for i in range(32):
            for j in range(32):
                reshape_x[0, i, j] = (x[i, j, 0] - 0) / 1
                reshape_x[1, i, j] = (x[i, j, 1] - 0) / 1
                reshape_x[2, i, j] = (x[i, j, 2] - 0) / 1
                '''
                reshape_x[0, i, j] = (x[i, j, 0] - 0.3403)/0.2724
                reshape_x[1, i, j] = (x[i, j, 1] - 0.3121)/0.2608
                reshape_x[2, i, j] = (x[i, j, 2] - 0.3214)/0.2669
                '''
        return reshape_x

    def stack_in_batch(self):
        dataset = []
        for batch in range(self.data_num//self.batch_size):
            data = np.zeros((self.batch_size, 3, 32, 32))
            classes = np.zeros((self.batch_size))
            for index in range(self.batch_size):
                _img, _class = self.get_item(batch * self.batch_size + index)
                data[index] = _img
                classes[index] = _class
            dataset.append((data, classes))
        random.shuffle(dataset)
        return dataset