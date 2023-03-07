import torch
import torch.nn as nn
import numpy as np
import time

class Model(nn.Module):
    def __init__(self, n_val, window, hidRNN):
        super(Model, self).__init__()
        self.P = window  # 输入窗口大小
        self.m = n_val  # 列数，变量数
        self.hidR = hidRNN
        self.GRU = nn.GRU(self.m, self.hidR)#100,32
        self.linear1 = nn.Linear(self.hidR, self.hidR)
        self.linear2 = nn.Linear(self.hidR, self.m)
    def forward(self, x):
        x1 = x.permute(1, 0, 2).contiguous()  # x1: [window, batch, n_val]
        a, h = self.GRU(x1)  # r: [1, batch, hidRNN]
        h = torch.squeeze(h, 0)  # r: [batch, hidRNN]
        res = torch.sigmoid(self.linear1(h))  # res: [batch, n_val]
        res = self.linear2(res)  # res: [batch, n_val]
        return res

