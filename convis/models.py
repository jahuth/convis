"""
Convis Models
--------------

These models are ready to run.

"""

from __future__ import print_function
import numpy as np
import uuid
from torch import nn
import torch

from .base import Layer
from .filters import Conv1d, Conv2d, Conv3d, TIME_DIMENSION, Delay, VariableDelay
from . import variables
from .retina import Retina

__all__ = ['L','LN','LNLN','LNFDLNF','LNFDSNF','McIntosh','Retina']

class L(Layer):
    """
        A linear model with a convolution filter.

        Pads input automatically to produce output of the same size as the input.
    """
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dims = 5
        super(L, self).__init__()
        in_channels = 1
        out_channels = 1
        if len(kernel_dim) == 5:
            in_channels = kernel_dim[1]
            out_channels = kernel_dim[0]
            kernel_dim = kernel_dim[2:]
        self.conv = Conv3d(in_channels, out_channels, kernel_dim, bias = bias, autopad=True, time_pad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv.bias.data[0] = 0.0
        self.conv.weight.data[:,:,:,:,:] = 0.0  
    def forward(self, x):
        return self.conv(x)


class LN(Layer):
    """
        A linear-nonlinear model with a convolution filter.

        Pads input automatically to produce output of the same size as the input.

        Parameters
        ----------
        kernel_dim: tuple(int,int,int) or tuple(int,int,int,int,int)
            Either the dimensions of a 3d kernel (time,x,y)
            or a 5d kernel (out_channels,in_channels,time,x,y).
        bias: bool
            Whether or not to include a scalar bias parameter in
            the linear filter
    """
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dims = 5
        self.nonlinearity = lambda x: x.clamp(min=0.0,max=1000000.0)
        super(LN, self).__init__()
        in_channels = 1
        out_channels = 1
        if len(kernel_dim) == 5:
            in_channels = kernel_dim[1]
            out_channels = kernel_dim[0]
            kernel_dim = kernel_dim[2:]
        self.conv = Conv3d(in_channels, out_channels, kernel_dim, bias = bias, autopad=True, time_pad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv.bias.data[0] = 0.0
        self.conv.weight.data[:,:,:,:,:] = 0.0  
    def forward(self, x):
        return self.nonlinearity(self.conv(x))


class LNLN(Layer):
    """
        A linear-nonlinear cascade model with two convolution filters.

        Pads input automatically to produce output of the same size as the input.
    """
    def __init__(self,kernel_dim=(1,1,1), bias = False):
        self.dims = 5
        self.nonlinearity = lambda x: x.clamp(min=0.0,max=1000000.0)
        super(LNLN, self).__init__()
        self.conv1 = Conv3d(1, 1, kernel_dim, bias = bias, autopad=True, time_pad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv1.bias.data[0] = 0.0
        self.conv2 = Conv3d(1, 1, kernel_dim, bias = bias, autopad=True, time_pad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv2.bias.data[0] = 0.0
    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        return x

LNSN = LNLN

class LNFDLNF(Layer):
    """
        A linear-nonlinear cascade model with two convolution filters,
        feedback for each layer and individual delays.

        Pads input automatically to produce output of the same size as the input.
    """
    def __init__(self,kernel_dim=(1,1,1), bias = False, feedback_length=50):
        self.dims = 5
        self.nonlinearity = lambda x: x.clamp(min=0.0)
        super(LNFDLNF, self).__init__()
        self.conv1 = Conv3d(1, 1, kernel_dim, bias = bias, time_pad=True, autopad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv1.bias.data[0] = 0.0
        self.conv2 = Conv3d(1, 1, kernel_dim, bias = bias, time_pad=True, autopad=True)
        if hasattr(self,'bias') and self.bias is not None:
            self.conv2.bias.data[0] = 0.0
        self.feedback1 = Conv3d(1,1,(feedback_length,1,1), time_pad=True, autopad=True)
        self.feedback2 = Conv3d(1,1,(feedback_length,1,1), time_pad=True, autopad=True)
        self.delay = VariableDelay()
    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x_feedback = self.feedback1(x)
        x = x - x_feedback
        x = self.delay(x)
        x = self.nonlinearity(self.conv2(x))
        x_feedback = self.feedback2(x)
        x = x - x_feedback
        return x

LNFDSNF = LNFDLNF

class McIntosh(Layer):
    """
        Convolutional Retina Model

        Contains two convolutional layers and one readout layer.

        The first convolutional layer has 8 channels.
        The second convolutional layer has 16 channels.
        The readout is a linear combination over all space
        and all channels of layer 2, resulting in `out_channels`
        many output channels.

        To set the weights individually::

            m = convis.models.McIntosh(out_channels=5)
            m.layer1.set_weight(np.random.randn(8,1,20,10,10))  # mapping from 1 to 8 channels
            m.layer2.set_weight(np.random.randn(16,8,10,50,50)) # mapping from 8 to 16 channels
            # the readout needs some number of outputs
            # and 16 x the number of pixels of the image as inputs
            m.readout.set_weight(np.random.randn(5,16*input.shape[-2]*input.shape[-1]))
            
            # plotting the parameters:
            m.plot()

        [1] Mcintosh, L. T., Maheswaranathan, N., Nayebi, A., Ganguli, S., & Baccus, S. A. (2016).
        Deep Learning Models of the Retinal Response to Natural Scenes.
        Advances in Neural Information Processing Systems 29 (NIPS), (Nips), 1-9.
        Also: arXiv:1702.01825 [q-bio.NC]
    """
    verbose = False
    def __init__(self,filter_size=(10,5,5), random_init=True, out_channels=1, filter_2_size=(1,1,1), layer1_channels = 8,
                layer2_channels = 16):
        super(McIntosh,self).__init__()
        layer1 = Conv3d(1,layer1_channels,filter_size,time_pad=True,autopad=True)
        self.add_module('layer1',layer1)
        self.layer1.set_weight(1.0,normalize=True)
        if random_init:
            self.layer1.set_weight(np.random.rand(layer1_channels,1,filter_size[0],filter_size[1],filter_size[2]),normalize=True)
        layer2 = Conv3d(layer1_channels,layer2_channels,filter_2_size,time_pad=True,autopad=True)
        self.add_module('layer2',layer2)
        self.layer2.set_weight(1.0,normalize=True)
        if random_init:
            self.layer2.set_weight(np.random.rand(layer2_channels,layer1_channels,filter_2_size[0],filter_2_size[1],filter_2_size[2]),normalize=True)
        self.readout = torch.nn.Linear(1,out_channels,bias=False)
    def forward(self, the_input):
        a = torch.nn.functional.relu(self.layer1(the_input))
        a = torch.nn.functional.relu(self.layer2(a))
        self._last_processed_image = a.size()
        # The readout should consider all channels and all locations
        # so we need to reshape the Tensor such that the 4th dimension
        # contains dimensions 1,3 and 4
        #  - moving dimension 3 to 4:
        a = torch.cat(a.split(1,dim=3),dim=4)
        #  - moving dimension 1 to 4:
        a = torch.cat(a.split(1,dim=1),dim=4)
        if self.readout.weight.size()[-1] != a.size()[-1]:
            if self.verbose:
                print('Resetting weight')
            if self._use_cuda:
                self.readout.weight = torch.nn.Parameter(torch.ones((self.readout.weight.size()[0],a.size()[-1])))
                self.readout.cuda()
            else:
                self.readout.weight = torch.nn.Parameter(torch.ones((self.readout.weight.size()[0],a.size()[-1])))
        a = self.readout(a)#.sum(dim=4,keepdim=True).sum(dim=3,keepdim=True).sum(dim=0,keepdim=True)
        return a
    def plot(self):
        from . import utils
        import matplotlib.pylab as plt
        utils.plot(self.layer1.weight)
        plt.title('Layer 1 bias:'+str(self.layer1.bias))
        utils.plot_5d_matshow(self.layer2.weight,dims=[(0,3),(1,2,4)])
        plt.title('Layer 2 bias:'+str(self.layer2.bias))
        if hasattr(self,'_last_processed_image'):
            utils.plot(self.readout.weight.squeeze().data.cpu().numpy().reshape((1,self._last_processed_image[-4],1,self._last_processed_image[-2],self._last_processed_image[-1])))
            plt.title('Readout')

