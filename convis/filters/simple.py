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
    def __init__(self,kernel_dim=(1,1,1),requires_grad=True):
        self.dim = 5
        super(TemporalLowPassFilterRecursive, self).__init__()
        #self.tau = Parameter(0.01,requires_grad=True)
        self.tau = torch.nn.Parameter(torch.Tensor([0.01]),requires_grad=requires_grad)
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
        self.last_y = y.detach()
        norm = 2.0*self.tau/steps#(self.tau/(self.tau+0.5))*steps
        return torch.cat(o,dim=TIME_DIMENSION)/norm


class TemporalHighPassFilterRecursive(Layer):
    def __init__(self,kernel_dim=(1,1,1),requires_grad=True):
        self.dim = 5
        super(TemporalHighPassFilterRecursive, self).__init__()
        #self.tau = Parameter(0.01,requires_grad=True)
        self.tau = torch.nn.Parameter(torch.Tensor([0.01]),requires_grad=requires_grad)
        self.k = torch.nn.Parameter(torch.Tensor([0.5]),requires_grad=requires_grad)
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


class SmoothConv(Layer):
    """
        A convolution with temporally smoothed filters.
        It can cover a long temporal period, but is a lot more
        efficient than a convlution filter of the same length.

        Each spatial filter `.g[n]` is applied to a temporally filtered
        signal with increasing delays by convolving multiple recursive
        exponential filters.

        The length of the filter depends on the number of temporal
        components and the time constant used for the delays.

        Each exponential filter `.e[n]` can have an individual 
        time constant, giving variable spacing between the filters.

        By default, the time constants are set to not create a gradient,
        so that they are not fittable.

        To show each component, use `get_all_components(some_input)`

        .. plot::
            :include-source:

            import matplotlib.pyplot as plt
            import numpy as np
            import convis
            s = convis.filters.simple.SmoothConv(n=6,tau=0.05)
            inp = np.zeros((1000,1,1))
            inp[50,0,0] = 1.0
            inp = convis.prepare_input(inp)
            c = s.get_all_components(inp)
            convis.plot_5d_time(c,mean=(3,4))
            c = c.data.cpu().numpy()



        Attributes
        ----------


        Methods
        -------


        See Also
        --------

        convis.filters.Conv3d : A full convolution layer 

    """
    def __init__(self,n=3,tau=0.1,spatial_filter=(10,10)):
        super(SmoothConv, self).__init__()
        self.dims=5
        self.e = []
        self.g = []
        for i in range(n):
            self.e.append(TemporalLowPassFilterRecursive(requires_grad=False))
            self.e[i].tau.data[0] = tau
            self.g.append(Conv3d(1,1,(1,spatial_filter[0],spatial_filter[1]),autopad=True, bias=False))
            self.g[i].set_weight(np.random.randn(1,spatial_filter[0],spatial_filter[1]))
        self.e = torch.nn.ModuleList(self.e)
        self.g = torch.nn.ModuleList(self.g)
    def forward(self,the_input):
        o = []
        y = the_input
        for i in range(len(self.e)):
            y = self.e[i](y)
            o.append(self.g[i](y))
        return torch.sum(torch.cat(o,dim=0),dim=0)[None,:,:,:,:]
    def get_all_components(self,the_input):
        o = []
        y = the_input
        for i in range(len(self.e)):
            y = self.e[i](y)
            o.append(self.g[i](y))
        return torch.cat(o,dim=1)