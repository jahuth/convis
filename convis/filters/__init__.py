import torch
from torch import nn
import numpy as np
import math
from .. import numerical_filters as nf
from ..base import Layer
TIME_DIMENSION = 2


class TimePadding(Layer):
    """
        Remembers references to previous time slices
        and prepends the input with `length` many
        time steps from previous calls.
        
        If the size of the image is changed without
        removing the state first, an Exception is
        raised.

        To avoid this, call `.clear_state()`. This method is recursive
        on all `convis.Layers`, so you only have to call it on the
        outermost `Layer`.
        If you want to store your history for one set of images,
        do some computation on other images and then return to
        the previous one, you can use `.push_state()` and `.pop_state()`.

    
    """
    def __init__(self,length=0):
        self.dim = 5
        self.length = length
        super(TimePadding, self).__init__()
        self.register_state('saved_inputs',[])
        self.saved_inputs = []
    @property
    def available_length(self):
        return np.sum([i.size()[TIME_DIMENSION] for i in self.saved_inputs])
    def forward(self, x):
        if len(self.saved_inputs) > 0:
            if x.size()[-2:] != self.saved_inputs[-1].size()[-2:]:
                raise Exception('input size does not match state size! Call `.clear_state()` on your model first!')
        while self.available_length < self.length:
            self.saved_inputs.append(x)
        if self._use_cuda:
            x_pad = torch.cat([i.cuda().detach() for i in self.saved_inputs] + [x.cuda()], dim=TIME_DIMENSION)
        else:
            x_pad = torch.cat([i.cpu().detach() for i in self.saved_inputs] +[x.cpu()], dim=TIME_DIMENSION)
        while self.available_length > self.length + x.size(TIME_DIMENSION):
            self.saved_inputs.pop(0)
        self.saved_inputs.append(x)
        return x_pad[:,:,-(self.length + x.size(TIME_DIMENSION)):,:,:]

class Delay(Layer):
    """
        Causes the input to be delayed by a set
        number of time steps.

            d = Delay(delay=100)
            d.run(some_input,10)

        Optionally, a length of input can also be prependet
        similar to the TimePadding Layer.
        
            d = Delay(delay=100,length=10) # additionally preprends 10 timesteps of each previous chunk
            d.run(some_input,10)

        When the size of the image is changed, the previous inputs
        do not match, so an Exception is raised.
        To avoid this, call `.clear_state()`. This method is recursive
        on all `convis.Layers`, so you only have to call it on the
        outermost `Layer`.
        If you want to store your history for one set of images,
        do some computation on other images and then return to
        the previous one, you can use `.push_state()` and `.pop_state()`.


    """
    def __init__(self,delay=0,length=0):
        self.dim = 5
        self.length = length
        self.delay = delay
        super(Delay, self).__init__()
        self.register_state('saved_inputs',[])
        self.saved_inputs = []
    @property
    def available_length(self):
        return np.sum([i.size()[TIME_DIMENSION] for i in self.saved_inputs])
    def forward(self, x):
        if len(self.saved_inputs) > 0:
            if x.size()[-2:] != self.saved_inputs[-1].size()[-2:]:
                raise Exception('input size does not match state size! Call `.clear_state()` on your model first!')
        while self.available_length < self.length + self.delay:
            self.saved_inputs.append(torch.zeros_like(x))
        if self._use_cuda:
            x_pad = torch.cat([i.cuda().detach() for i in self.saved_inputs] + [x.cuda()], dim=TIME_DIMENSION)
        else:
            x_pad = torch.cat([i.cpu().detach() for i in self.saved_inputs] +[x.cpu()], dim=TIME_DIMENSION)
        while self.available_length > self.length + x.size(TIME_DIMENSION) + self.delay:
            self.saved_inputs.pop(0)
        self.saved_inputs.append(x)
        to = -self.delay if self.delay > 0 else None
        return x_pad[:,:,-(self.length + x.size(TIME_DIMENSION) + self.delay):to,:,:]

class VariableDelay(Layer):
    """
        This Layer applies variable delays to each 
        pixel of the input.
    
        Example::

            
            v = VariableDelay(delays = d)

        At the moment, the delays do *not* provide a gradient.

        Possible future feature if requested:
        variable delay per pixel, channel and batch dimension.
    """
    def __init__(self, delays = None):
        super(VariableDelay, self).__init__()
        if delays is None:
            delays = torch.zeros((1,1,1,1,1))
        self.delays = torch.nn.Parameter(delays)
        self.all_delay = Delay()
    def forward(self, x):
        if self.delays is None or self.delays.size()[-2:] != x.size()[-2:]:
            self.delays = torch.nn.Parameter(torch.ones(x.size()[-2:]))
        self.all_delay.delay = int(torch.min(self.delays))
        self.all_delay.length = int(torch.max(self.delays)-torch.min(self.delays))
        x_delayed = self.all_delay(x)
        ind_to = -(self.delays - int(torch.min(self.delays)))
        ind_from = -(-ind_to + x.size()[2])
        x_out = []
        for i,x_row in enumerate(x_delayed.split(1,-1)):
            new_row = []
            for j,x_pixel in enumerate(x_row.split(1,-2)):
                #print int(ind_from[i,j]),int(ind_to[i,j])
                to = int(ind_to[i,j])
                if to == 0:
                    to = None
                new_row.append(x_pixel[:,:,int(ind_from[i,j]):to,:,:])
            x_out.append(torch.cat(new_row,dim=-2))
        x_out = torch.cat(x_out,dim=-1)
        return x_out

class Conv3d(torch.nn.Conv3d):
    """
        Does a convolution, but pads the input in time
        with previous input and in space by replicating
        the edge.

        Arguments:

            * in_channels
            * out_channels
            * kernel_size
            * bias (bool)

        Additional PyTorch Conv3d keyword arguments:

            * padding (should not be used)
            * stride
            * dilation
            * groups

        Additional convis Conv3d keyword arguments:

            * time_pad: False (enables padding in time)
            * autopad: False (enables padding in space)

        To change the weight, use the method `set_weight()`
        which also accepts numpy arguments.


    """
    def __init__(self,in_channels=1,out_channels=1,kernel_size=(1,1,1),bias=True,*args,**kwargs):
        self.do_adjust_padding = kwargs.get('adjust_padding',False)
        self.do_time_pad = kwargs.get('time_pad',False)
        self.autopad = kwargs.get('autopad',False)
        self.autopad_mode = 'replicate'
        if 'adjust_padding' in kwargs.keys():
            del kwargs['adjust_padding']
        if 'time_pad' in kwargs.keys():
            del kwargs['time_pad']
        if 'autopad' in kwargs.keys():
            del kwargs['autopad']
        super(Conv3d, self).__init__(in_channels,out_channels,kernel_size,bias=bias*args,**kwargs)
        if hasattr(self,'bias') and self.bias is not None:
            self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
        self.time_pad = TimePadding(self.weight.size()[TIME_DIMENSION])
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
    def filter_length(self):
        return self.weight.data.shape[TIME_DIMENSION] - 1
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
        if self.do_time_pad:
            self.time_pad.length = self.filter_length
            x = self.time_pad(x)
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
