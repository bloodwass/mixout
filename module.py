##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Cheolhyoung Lee
## Department of Mathematical Sciences, KAIST
## Email: cheolhyoung.lee@kaist.ac.kr
## Implementation of mixout from https://arxiv.org/abs/1909.11299
## "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn import Parameter

from mixout import mixout

class MixLinear(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']
    # If target is None, nn.Sequential(nn.Linear(m, n), MixLinear(m', n', p)) 
    # is equivalent to nn.Sequential(nn.Linear(m, n), nn.Dropout(p), nn.Linear(m', n')).
    # If you want to change a dropout layer to a mixout layer, 
    # you should replace nn.Linear right after nn.Dropout(p) with Mixout(p) 
    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.target = target
        self.p = p
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        return F.linear(input, mixout(self.weight, self.target, 
                                      self.p, self.training), self.bias)

    def extra_repr(self):
        type = 'drop' if self.target is None else 'mix' 
        return '{}={}, in_features={}, out_features={}, bias={}'.format(type+"out", self.p,
            self.in_features, self.out_features, self.bias is not None)