import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class PosLinear(nn.Linear):
    """
    权重保持为正的全连接层
    """
    def forward(self, input):
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
    