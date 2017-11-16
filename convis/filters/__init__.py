import torch
from torch import nn
import numpy as np
import math
from .. import numerical_filters as nf
TIME_DIMENSION = 2

class Conv3d(nn.Conv3d):
    def __init__(self,*args,**kwargs):
        self.do_adjust_padding = kwargs.get('adjust_padding',False)
        self.autopad = kwargs.get('autopad',False)
        self.autopad_mode = 'replicate'
        if 'adjust_padding' in kwargs.keys():
            del kwargs['adjust_padding']
        if 'autopad' in kwargs.keys():
            del kwargs['autopad']
        super(Conv3d, self).__init__(*args,**kwargs)
        if hasattr(self,'bias') and self.bias is not None:
            self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
    def set_weight(self,w,normalize=False):
        if type(w) in [int,float]:
            self.weight.data = torch.ones(self.weight.data.shape) * w
        else:
            if len(w.shape) == 1:
                w = w[None,None,:,None,None]
            if len(w.shape) == 2:
                w = w[None,None,None,:,:]
            if len(w.shape) == 3:
                w = w[None,None,:,:,:]
            self.weight.data = torch.Tensor(w)
            self.kernel_size = self.weight.data.shape[2:]
        if normalize:
            self.weight.data = self.weight.data / self.weight.data.sum()
        if self.do_adjust_padding:
            self.adjust_padding()
    def adjust_padding(self):
        self.padding = (int(math.ceil((self.kernel_size[0])/2)),
                        int(math.ceil((self.kernel_size[1])/2)),
                        int(math.ceil((self.kernel_size[1])/2)))
    @property
    def kernel_padding(self):
        k = np.array(self.weight.data.shape[2:])
        return (int(math.floor((k[2])/2.0))-1,
                int(math.ceil(k[2]))-int(math.floor((k[2])/2.0)),
                int(math.floor((k[1])/2.0))-1,
                int(math.ceil(k[1]))-int(math.floor((k[1])/2.0)),
                0,0)
    @property
    def kernel_padding_all(self):
        k = np.array(self.weight.data.shape[2:])
        return (int(math.floor((k[2])/2.0))-1,
                int(math.ceil(k[2]))-int(math.floor((k[2])/2.0)),
                int(math.floor((k[1])/2.0))-1,
                int(math.ceil(k[1]))-int(math.floor((k[1])/2.0)),
                int(math.floor((k[0])/2.0))-1,
                int(math.ceil(k[0]))-int(math.floor((k[0])/2)))
    def exponential(self,adjust_padding=False,*args,**kwargs):
        self.set_weight(nf.exponential_filter_1d(*args,**kwargs)[::-1],normalize=False)
        if adjust_padding:
            self.adjust_padding()
    def highpass_exponential(self,adjust_padding=False,*args,**kwargs):
        self.set_weight(nf.exponential_highpass_filter_1d(*args,**kwargs)[::-1],normalize=False)
        if adjust_padding:
            self.adjust_padding()
    def gaussian(self,sig,adjust_padding=False):
        self.set_weight(nf.gauss_filter_5d(sig,sig),normalize=False)
        if adjust_padding:
            self.adjust_padding()
    def __len__(self):
        return self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]
    def forward(self,x):
        if self.autopad:
            x = torch.nn.functional.pad(x,self.kernel_padding, self.autopad_mode)
        return super(Conv3d, self).forward(x)

class Conv2d(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2d, self).__init__(*args,**kwargs)
        if hasattr(self,'bias') and self.bias is not None:
            self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
    def set_weight(self,w,normalize=False):
        if type(w) in [int,float]:
            self.weight.data = torch.ones(self.weight.data.shape) * w
        else:
            if self.weight.data.shape == w.shape:
                self.weight.data = torch.Tensor(w)
            else:
                if len(w.shape) == 4:
                    w_h = w.shape[2]
                    w_w = w.shape[3]
                    self.weight.data[0,0,:w_h,:w_w] = torch.Tensor(w[0,0])
                else:
                    w_h = w.shape[0]
                    w_w = w.shape[1]
                    self.weight.data[0,0,:w_h,:w_w] = torch.Tensor(w)
        if normalize:
            self.weight.data = self.weight.data / self.weight.data.sum()
    def gaussian(self,sig):
        self.set_weight(nf.gauss_filter_2d(sig,sig)[None,None,:,:],normalize=False)
        
class Conv1d(nn.Conv1d):
    def __init__(self,*args,**kwargs):
        super(Conv1d, self).__init__(*args,**kwargs)
        if hasattr(self,'bias') and self.bias is not None:
            self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
    def set_weight(self,w,normalize=False):
        if type(w) in [int,float]:
            self.weight.data = torch.ones(self.weight.data.shape) * w
        else:
            self.weight.data = torch.Tensor(w)
        if normalize:
            self.weight.data = self.weight.data / self.weight.data.sum()
    def exponential(self,*args,**kwargs):
        self.set_weight(nf.exponential_filter_1d(*args,**kwargs),normalize=False)