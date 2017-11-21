import numpy as np
import uuid
from torch import nn
import torch

from ..base import Layer
from ..filters import Conv1d, Conv2d, Conv3d, TIME_DIMENSION
from .. import variables

class L(Layer):
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dim = 5
        super(L, self).__init__()
        self.conv = Conv3d(1, 1, kernel_dim, bias = bias)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv.bias.data[0] = 0.0
        self.conv.weight.data[:,:,:,:,:] = 0.0  
    def forward(self, x):
        return self.conv(x)


class LN(Layer):
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dim = 5
        super(LN, self).__init__()
        self.conv = Conv3d(1, 1, kernel_dim, bias = bias)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv.bias.data[0] = 0.0
        self.conv.weight.data[:,:,:,:,:] = 0.0  
    def forward(self, x):
        return self.conv(x).clamp(min=0.0)

class TemporalLowPassFilterRecursive(Layer):
    def __init__(self,kernel_dim=(1,1,1)):
        self.dim = 5
        super(TemporalLowPassFilterRecursive, self).__init__()
        #self.tau = Parameter(0.01,requires_grad=True)
        self.tau = torch.nn.Parameter(torch.Tensor([0.01]),requires_grad=True)
    def clear(self):
        if hasattr(self,'last_y'):
            del self.last_y
    def forward(self, x):
        steps = variables.Parameter(1.0/variables.default_resolution.steps_per_second,requires_grad=False)
        if self._use_cuda:
            steps = steps.cuda()
        a_0 = 1.0
        a_1 = -torch.exp(-steps/self.tau)
        b_0 = 1.0 - a_1
        if hasattr(self,'last_y'):
            y = self.last_y
        else:
            y = torch.autograd.Variable(torch.zeros(1,1,1,x.data.shape[3],x.data.shape[4]))
            if self._use_cuda:
                y = y.cuda()
        o = []
        for i in range(x.data.shape[TIME_DIMENSION]):
            y = (x[:,:,i,:,:] * b_0 - y * a_1) / a_0
            o.append(y)
        self.last_y = y
        norm = 2.0*self.tau/steps#(self.tau/(self.tau+0.5))*steps
        return torch.cat(o,dim=TIME_DIMENSION)/norm


class TemporalHighPassFilterRecursive(Layer):
    def __init__(self,kernel_dim=(1,1,1)):
        self.dim = 5
        super(TemporalHighPassFilterRecursive, self).__init__()
        #self.tau = Parameter(0.01,requires_grad=True)
        self.tau = torch.nn.Parameter(torch.Tensor([0.01]),requires_grad=True)
        self.k = torch.nn.Parameter(torch.Tensor([0.5]),requires_grad=True)
    def clear(self):
        if hasattr(self,'last_y'):
            del self.last_y
    def forward(self, x):
        steps = variables.Parameter(1.0/variables.default_resolution.steps_per_second,requires_grad=False)
        if self._use_cuda:
            steps = steps.cuda()
        a_0 = 1.0
        a_1 = -torch.exp(-steps/self.tau)
        b_0 = 1.0 - a_1
        if hasattr(self,'last_y'):
            y = self.last_y
        else:
            y = torch.autograd.Variable(torch.zeros(1,1,1,x.data.shape[3],x.data.shape[4]))
            if self._use_cuda:
                y = y.cuda()
        o = []
        x1 = x[:,:,0,:,:] 
        for i in range(x.data.shape[TIME_DIMENSION]):
            y = (x1 * b_0 - y * a_1) / a_0
            x1 = x[:,:,i,:,:] 
            o.append(y)
        self.last_y = y
        norm = 2.0*self.tau/steps#(self.tau/(self.tau+0.5))*steps
        return x - (self.k)*torch.cat(o,dim=TIME_DIMENSION)/norm

