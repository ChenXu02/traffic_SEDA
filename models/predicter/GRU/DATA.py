import torch
import numpy as np
from torch.autograd import Variable
from csv import reader
import random

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))
class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name1, train, valid, horizon, window, normalize=3):
        self.P = window
        self.h = horizon
        self.std = 0
        self.mean = 0
        self.max=0
        self.rawdat = np.load(file_name1)['data'][:, :, 0]
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self.validstartnum=int(train * self.n)
        self.teststartnum=int((train + valid) * self.n)
        self._split(int(train * self.n), int((train + valid) * self.n))
        self.scale = torch.from_numpy(self.scale).float()



    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat
        if (normalize == 1):
            self.dat = self.rawdat / (np.max(self.rawdat))
            self.max=np.max(self.rawdat)
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / (np.max(np.abs(self.rawdat[:, i])))
        if (normalize==3):
            self.std=np.std(self.rawdat)
            self.mean=np.mean(self.rawdat)
            self.dat=(self.rawdat-self.mean)/self.std

    def _split(self, train, valid):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.trainstartnum=self.P + self.h - 1
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - horizon + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]








