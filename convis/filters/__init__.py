import torch
from torch import nn
from .. import numerical_filters as nf
TIME_DIMENSION = 2

class Conv3d(nn.Conv3d):
    def __init__(self,*args,**kwargs):
        super(Conv3d, self).__init__(*args,**kwargs)
        self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
    def set_weight(self,w,normalize=True):
        if type(w) in [int,float]:
            self.weight.data = torch.ones(self.weight.data.shape) * w
        else:
            self.weight.data = torch.Tensor(w)
        if normalize:
            self.weight.data = self.weight.data / self.weight.data.sum()
    def exponential(self,*args,**kwargs):
        self.set_weight(nf.exponential_filter_5d(*args,**kwargs),normalize=False)
    def gaussian(self,sig):
        self.set_weight(nf.gauss_filter_5d(sig,sig),normalize=False)

class Conv2d(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2d, self).__init__(*args,**kwargs)
        self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
    def set_weight(self,w,normalize=True):
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
        self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
    def set_weight(self,w,normalize=True):
        if type(w) in [int,float]:
            self.weight.data = torch.ones(self.weight.data.shape) * w
        else:
            self.weight.data = torch.Tensor(w)
        if normalize:
            self.weight.data = self.weight.data / self.weight.data.sum()
    def exponential(self,*args,**kwargs):
        self.set_weight(nf.exponential_filter_1d(*args,**kwargs),normalize=False)