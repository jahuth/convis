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
        self.input_state = torch.autograd.Variable(torch.zeros((1,1,1,1,1)))
    @property
    def filter_length(self):
        return self.conv.weight.data.shape[TIME_DIMENSION] - 1
    def forward(self, x):
        if not (self.input_state.data.shape[TIME_DIMENSION] == self.filter_length and
            self.input_state.data.shape[3] == x.data.shape[3] and
            self.input_state.data.shape[4] == x.data.shape[4]):
            self.input_state = torch.autograd.Variable(torch.zeros((x.data.shape[0],x.data.shape[1],self.filter_length,x.data.shape[3],x.data.shape[4])))
            x_init = x[:,:,:self.filter_length,:,:]
            self.input_state[:,:,(-x_init.data.shape[2]):,:,:] = x_init
        if self._use_cuda:
            self.input_state = self.input_state.cuda()
            x_pad = torch.cat([self.input_state.cuda(), x.cuda()], dim=TIME_DIMENSION)
        else:
            self.input_state = self.input_state.cpu()
            #print self.input_state, x
            x_pad = torch.cat([self.input_state.cpu(), x.cpu()], dim=TIME_DIMENSION)
        self.input_state = x_pad[:,:,-(self.filter_length):,:,:]
        return self.conv(x_pad)


class LN(Layer):
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dims = 5
        self.nonlinearity = lambda x: x.clamp(min=0.0)
        super(LN, self).__init__()
        self.conv = Conv3d(1, 1, kernel_dim, bias = bias, autopad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv.bias.data[0] = 0.0
        self.conv.weight.data[:,:,:,:,:] = 0.0  
    def forward(self, x):
        return self.nonlinearity(self.conv(x))


class LNLN(Layer):
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dims = 5
        self.nonlinearity = lambda x: x.clamp(min=0.0)
        super(LNLN, self).__init__()
        self.conv1 = Conv3d(1, 1, kernel_dim, bias = bias, autopad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv1.bias.data[0] = 0.0
        self.conv2 = Conv3d(1, 1, kernel_dim, bias = bias, autopad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv2.bias.data[0] = 0.0
    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        return x


