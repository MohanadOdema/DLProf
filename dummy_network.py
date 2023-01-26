# code to get the flops (MACs) using fvcore, needed for estimating the operational intensity of each layer.

import torch
import torch.nn.functional as F
from torch import nn
# from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
import nvidia_dlprof_pytorch_nvtx as nvtx
nvtx.init()

import numpy as np

from torch.backends import cudnn

class dummy_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = (self.fc1(x))
        x = (self.fc2(x))
        x = self.fc3(x)
        return x
    

network = dummy_network().cuda()
input = torch.randn(1,3,32,32).cuda()

if torch.cuda.is_available():
    cudnn.benchmark = True

with torch.autograd.profiler.emit_nvtx():
    output = network(input)


# flops = FlopCountAnalysis(network, input)
# handlers = {"aten::pool": None}             # add layers to ignore their usage
# flops.set_op_handle(**handlers)             # ignore max_pool as fvcore does not work with it. It does not have MACs anyway
# print(flop_count_table(flops))