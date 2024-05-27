import torchvision
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp(nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(mlp, self).__init__()

        # 定义输入到隐藏层的线性层
        self.linear1 = nn.Linear(num_i, num_h)
        self.bn1 = nn.BatchNorm1d(num_h)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)

        # 定义隐藏层到隐藏层的线性层
        self.linear2 = nn.Linear(num_h, num_h)
        self.bn2 = nn.BatchNorm1d(num_h)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)

        # 定义隐藏层到输出层的线性层
        self.linear3 = nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear3(x)
        return F.softmax(x)
