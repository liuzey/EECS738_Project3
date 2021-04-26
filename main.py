import numpy as np
import os
import time
import sys
import copy
import argparse
from datasets import GDataLoader
from MLP import MLP
from CNN import CNN
from mnist import MNIST

LEARNING_RATE = 1e-3
BATCH_SIZE = 2048
N_EPOCH = 1000
LOG_INTERVAL = 10
CLASS_NUM = {'mnist': 10, 'gtsrb': 43}

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help='Dataset. Choose from GTSRB and MNIST')
parser.add_argument("-p", type=bool, default=False, help='Whether to load pre-trained parameters.')
parser.add_argument("-s", type=bool, default=False, help='Whether to save trained parameters.')
args = parser.parse_args()


def load_data(_train=True, dataname='gtsrb'):
    if dataname == 'mnist':
        mndata = MNIST('./mnist/raw/')
        mndata.gz = False
        if _train:
            images, labels = mndata.load_training()
        else:
            images, labels = mndata.load_testing()
        images = np.array(images)
        labels = np.array(labels)
        dataloader = [(images[i*BATCH_SIZE:(i+1)*BATCH_SIZE], labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                      for i in range(images.shape[0]//BATCH_SIZE)]
    elif dataname == 'gtsrb':
        data = GDataLoader('./GTSRB/', train, BATCH_SIZE)
        dataloader = data.stack_in_batch()
    else:
        print('Dataset not found. Check inputs.')
        exit()
    print('Data successfully loaded.')
    return dataloader


def train(model, dataloader):
    model.train = True
    for epoch in range(1, N_EPOCH + 1):
        len_dataloader = len(dataloader)
        for i in range(len_dataloader):
            loss_class = 0
            pack = dataloader[i]
            img, label = pack[0], pack[1]
            class_output = model.forward(img)
            err, loss_c = model.loss_(class_output, label)
            pred = np.argmax(class_output, axis=1)
            # print(pred.reshape(-1))
            acc = (pred[1] == label).sum()/BATCH_SIZE * 100
            # print(class_output[0])
            loss_class += np.sum(np.sum(loss_c, axis=0), axis=0)/BATCH_SIZE
            model.backward(err)

            if i % LOG_INTERVAL == 0:
                print('Epoch: [{}/{}], Batch: [{}/{}], Loss: {:.4f}, Acc: {}%'.format(epoch, N_EPOCH, i, len_dataloader, loss_class, acc))

            if args.s and i % (1 * LOG_INTERVAL) == 0:
                model.save('./paras/')


def test(model, dataloader):
    model.train = True
    n_correct = 0
    len_dataloader = len(dataloader)
    for i in range(len_dataloader):
        pack = dataloader[i]
        img, label = pack[0], pack[1]
        class_output = model.forward(img)
        pred = np.argmax(class_output, 1)
        n_correct += (pred[1] == label).sum()
        # print(n_correct)

    accu = float(n_correct) / (len(dataloader)*BATCH_SIZE) * 100
    print('Accuracy on {} dataset: {:.4f}%'.format(args.data, accu))
    return accu


if __name__ == '__main__':
    train_data = load_data(_train=True, dataname=args.data)
    test_data = load_data(_train=False, dataname=args.data)
    if args.data == 'mnist':
        model = MLP(CLASS_NUM[args.data])
        if args.p:
            model.load('./paras/')
    elif args.data == 'gtsrb':
        model = CNN(CLASS_NUM[args.data])
        if args.p:
            model.load('./paras_save/')
    # train(model, train_data)
    _ = test(model, test_data)