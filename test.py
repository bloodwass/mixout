import torch
from torch import nn
from mixout import MixLinear

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = MixLinear(in_features=1, out_features=1, p=0.4)