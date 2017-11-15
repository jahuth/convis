import numpy as np
import uuid
from torch import nn
import torch

from .base import Layer
from .filters import Conv1d, Conv2d, Conv3d, TIME_DIMENSION
from . import variables

class L(Layer):
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dims = 5
        super(L, self).__init__()
        self.conv = Conv3d(1, 1, kernel_dim, bias = bias, autopad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv.bias.data[0] = 0.0
        self.conv.weight.data[:,:,:,:,:] = 0.0  
    def forward(self, x):
        return self.conv(x)


class LN(Layer):
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dims = 5
        super(LN, self).__init__()
        self.conv = Conv3d(1, 1, kernel_dim, bias = bias, autopad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv.bias.data[0] = 0.0
        self.conv.weight.data[:,:,:,:,:] = 0.0  
    def forward(self, x):
        return self.conv(x).clamp(min=0.0)


class LNLN(Layer):
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dims = 5
        super(LN, self).__init__()
        self.conv1 = Conv3d(1, 1, kernel_dim, bias = bias, autopad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv1.bias.data[0] = 0.0
        self.conv2 = Conv3d(1, 1, kernel_dim, bias = bias, autopad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv2.bias.data[0] = 0.0
    def forward(self, x):
        x = self.conv1(x).clamp(min=0.0)
        x = self.conv2(x).clamp(min=0.0)
        return x


