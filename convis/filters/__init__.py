import torch
from torch import nn
import numpy as np
import math
from .. import numerical_filters as nf
from .. import variables
from ..base import Layer
TIME_DIMENSION = 2
X_DIMENSION = 3
Y_DIMENSION = 4


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
                raise Exception('input size '+str(x.size()[-2:])+' does not match state size ('+str(self.saved_inputs[-1].size()[-2:])+')! Call `.clear_state()` on your model first!')
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

class Conv3d(torch.nn.Conv3d,Layer):
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
        super(Conv3d, self).__init__(in_channels,out_channels,kernel_size,bias=bias,*args,**kwargs)
        if hasattr(self,'bias') and self.bias is not None:
            self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
        self.time_pad = TimePadding(self.weight.size()[TIME_DIMENSION])
    def set_weight(self,w,normalize=False,preserve_channels=False):
        """
            Sets a new weight for the convolution.

            Parameters
            ----------
            w: numpy array or PyTorch Tensor
                The new kernel `w` should have 1,2,3 or 5 dimensions.
                    1 dimensions: temporal kernel
                    2 dimensions: spatial kernel
                    3 dimensions: spatio-temporal kernel (time,x,y)
                    5 dimensions: spatio-temporal kernels for multiple channels
                        (out_channels, in_channels, time, x, y)
                If the new kernel has 1, 2 or 3 dimensions and 
                `preserve_channels` is `True`, the input and output 
                channels will be preserved and the same kernel
                will be applied to all channel combinations.
                (ie. each output channel recieves the sum of all
                input channels).
                This makes sense if the kernel is further optimized,
                otherwise, the same effect can be achieved with a 
                single input and output channel more effectively.

            normalize: bool (default: False)
                Whether or not the sum of the kernel values
                should be normalized to 1, such that the
                sum over all input values and all output 
                values is the approximately same.

            preserve_channels: bool (default: False)
                Whether or not to copy smaller kernels
                to all input-output channel combinations.

        """
        if type(w) in [int,float]:
            self.weight.data = torch.ones(self.weight.data.shape) * w
        else:
            if len(w.shape) == 5:
                self.out_channels = w.shape[0]
                self.in_channels = w.shape[1]
            else:
                if len(w.shape) == 1:
                    w = w[None,None,:,None,None]
                elif len(w.shape) == 2:
                    w = w[None,None,None,:,:]
                elif len(w.shape) == 3:
                    w = w[None,None,:,:,:]
                if preserve_channels is True:
                    w = w * np.ones((self.out_channels, self.in_channels, 1, 1, 1))
            self.weight.data = torch.Tensor(w)
            if self._use_cuda:
                self.weight.data = self.weight.data.cuda()
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
        self.set_weight(nf.exponential_filter_1d(*args,**kwargs)[::-1].copy(),normalize=False)
        if adjust_padding:
            self.adjust_padding()
    def highpass_exponential(self,adjust_padding=False,*args,**kwargs):
        self.set_weight(nf.exponential_highpass_filter_1d(*args,**kwargs)[::-1].copy(),normalize=False)
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

class RF(Conv3d):
    """
        A Receptive Field Layer

        Does a convolution and pads the input in time
        with previous input, just like Conv3d, but with
        no spatial padding, resulting in a single output
        pixel.

        To use it correctly, the weight should be set to 
        the same spatial dimensions as the input.
        However, if the weight is larger than the input
        or the input is larger than the weight,
        the input is padded or cut. The parameter `rf_mode`
        controls the placement of the receptive field
        on the image.

        Currently, only rf_mode='corner' is implemented,
        which keeps the top left pixel identical and only
        extends or cuts the right and bottom portions
        of the input.

        .. warning::

            The spatial extent of your weight should match your input images to 
            get meaningful receptive fields. Otherwise the receptive field is placed
            at the top left corner of the input.

            If the weight was not set manually, the first time the filter sees input
            it creates an empty weight of the matching size. However when
            the input size is changed, the weight does not change automatically
            to match new input. Use :meth:`reset_weight()` to reset the weight
            or change the size manually.

            Any receptive field of size 1 by 1 pixel is considered
            empty and will be replaced with a uniform
            weight of the size of the input the next time
            the filter is used.


        Examples
        --------

            >>> m = convis.filters.RF()
            >>> inp = convis.samples.moving_gratings()
            >>> o = m.run(inp, dt=200)
            >>> o.plot()

        Or as a part of a cascade model::

            >>> m = convis.models.LNCascade()
            >>> m.add_layer(convis.filters.Conv3d(1,5,(1,10,10)))
            >>> m.add_layer(convis.filters.RF(5,1,(10,1,1)))
                # this RF will take into account 10 timesteps, it's width and height will be set by the input
            >>> inp = convis.samples.moving_grating()
            >>> o = m.run(inp, dt=200)

        See Also
        --------
        Conv3d
    """
    def __init__(self,in_channels=1,out_channels=1,kernel_size=(1,1,1),bias=True,rf_mode='corner',*args,**kwargs):
        autopad = kwargs.get('autopad',False)
        kwargs['autopad'] = autopad
        self.rf_placement_mode = rf_mode
        super(RF, self).__init__(in_channels,out_channels,kernel_size,bias=bias, *args,**kwargs)
    def reset_weight(self):
        self.set_weight(np.zeros((self.weight.size()[2],1,1)))
    def forward(self,x):
        if self.weight.size()[3] == 1 and self.weight.size()[4] == 1:
            self.set_weight(np.ones((self.weight.size()[2],x.size()[3],x.size()[4])),normalize=True)
        if self.do_time_pad:
            self.time_pad.length = self.filter_length
            x = self.time_pad(x)
        if self.rf_placement_mode is 'corner':
            if x.size()[3] < self.weight.size()[3]:
                x = torch.nn.functional.pad(x,(0,0,0,self.weight.size()[3],0,0))
            if x.size()[4] < self.weight.size()[4]:
                x = torch.nn.functional.pad(x,(0,self.weight.size()[4],0,0,0,0))
            if x.size()[3] > self.weight.size()[3]:
                x = x[:,:,:,:self.weight.size()[3],:]
            if x.size()[4] > self.weight.size()[4]:
                x = x[:,:,:,:,:self.weight.size()[4]]
        else:
            raise Exception('RF placements other than \'corner\' are not implemented yet!')
        return super(RF, self).forward(x)

class Conv2d(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        self.autopad = kwargs.get('autopad',False)
        self.autopad_mode = 'replicate'
        if 'autopad' in kwargs.keys():
            del kwargs['autopad']
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
                    self.out_channels = w.shape[0]
                    self.in_channels = w.shape[1]
                    w_h = w.shape[2]
                    w_w = w.shape[3]
                    self.weight.data = torch.Tensor(w)
                else:
                    w_h = w.shape[0]
                    w_w = w.shape[1]
                    self.weight.data = torch.Tensor(w)[None,None]
        if normalize:
            self.weight.data = self.weight.data / self.weight.data.sum()
    @property
    def kernel_padding(self):
        k = np.array(self.weight.data.shape[-2:])
        return (int(math.floor((k[1])/2.0))-1,
                int(math.ceil(k[1]))-int(math.floor((k[1])/2.0)),
                int(math.floor((k[0])/2.0))-1,
                int(math.ceil(k[0]))-int(math.floor((k[0])/2.0)))
    def forward(self,x):
        if self.autopad:
            x = torch.nn.functional.pad(x,self.kernel_padding, self.autopad_mode)
        return super(Conv2d, self).forward(x)
    def gaussian(self,sig):
        self.set_weight(nf.gauss_filter_2d(sig,sig)[None,None,:,:],normalize=False)
        
class Conv1d(nn.Conv1d):
    def __init__(self,*args,**kwargs):
        self.do_time_pad = kwargs.get('time_pad',False)
        if 'time_pad' in kwargs.keys():
            del kwargs['time_pad']
        super(Conv1d, self).__init__(*args,**kwargs)
        if hasattr(self,'bias') and self.bias is not None:
            self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
    @property
    def filter_length(self):
        return self.weight.data.shape[0]
    def set_weight(self,w,normalize=False):
        if type(w) in [int,float]:
            self.weight.data = torch.ones(self.weight.data.shape) * w
        else:
            self.weight.data = torch.Tensor(w)
        if normalize:
            self.weight.data = self.weight.data / self.weight.data.sum()
    def forward(self,x):
        if self.do_time_pad:
            self.time_pad.length = self.filter_length
            x = self.time_pad(x)
        return super(Conv3d, self).forward(x)
    def exponential(self,*args,**kwargs):
        self.set_weight(nf.exponential_filter_1d(*args,**kwargs),normalize=False)

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
        self.register_state('last_y',None)
    def clear(self):
        if hasattr(self,'last_y'):
            self.last_y = None
    def forward(self, x):
        steps = variables.Parameter(1.0/variables.default_resolution.steps_per_second,requires_grad=False)
        if self._use_cuda:
            steps = steps.cuda()
        a_0 = 1.0
        a_1 = -torch.exp(-steps/self.tau)
        b_0 = 1.0 - a_1
        if self.last_y is not None:
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
        self.register_state('last_y',None)
    def clear(self):
        if hasattr(self,'last_y'):
            self.last_y = None
    def forward(self, x):
        steps = variables.Parameter(1.0/variables.default_resolution.steps_per_second,requires_grad=False)
        if self._use_cuda:
            steps = steps.cuda()
        a_0 = 1.0
        a_1 = -torch.exp(-steps/self.tau)
        b_0 = 1.0 - a_1
        if self.last_y is not None:
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

def _select_(x,dim,i):
    if dim == 0:
        return x[i,:,:,:,:,][None,:,:,:,:]
    if dim == 1:
        return x[:,i,:,:,:][:,None,:,:,:]
    if dim == 2:
        return x[:,:,i,:,:][:,:,None,:,:]
    if dim == 3:
        return x[:,:,:,i,:][:,:,:,None,:]
    if dim == 4:
        return x[:,:,:,:,i][:,:,:,:,None]

class SpatialRecursiveFilter(Layer): 
    def __init__(self,kernel_dim=(1,1,1),requires_grad=True):
        self.dim = 5
        super(SpatialRecursiveFilter, self).__init__()
        self.density = torch.nn.Parameter(torch.Tensor([1.0]))
    def forward(self, x):
        config = {}
        alpha = 1.695 * self.density
        ema = torch.exp(-alpha)
        ek = (1.0-ema)*(1.0-ema) / (1.0+2.0*alpha*ema - ema*ema)
        A1 = ek
        A2 = ek * ema * (alpha-1.0)
        A3 = ek * ema * (alpha+1.0)
        A4 = -ek*ema*ema
        B1 = 2.0*ema
        B2 = -ema*ema
        def smooth_forward(x,a1,a2,b1,b2,dim):
            x1 = _select_(x,dim,0)
            o = []
            y1 = torch.autograd.Variable(torch.zeros_like(x1.data))
            y2 = torch.autograd.Variable(torch.zeros_like(x1.data))
            x2 = torch.autograd.Variable(torch.zeros_like(x1.data))
            for i in range(x.data.shape[dim]):
                x1,x2 = _select_(x,dim,i),x1
                y = (a1 * x1 + a2 * x2 + b1 * y1 + b2 * y2)
                y1, y2 = y, y1
                o.append(y)
            o = torch.cat(o,dim=dim)
            return o
        def smooth_backward(x,a1,a2,b1,b2,dim):
            x1 = _select_(x,dim,0)
            o = []
            y1 = torch.autograd.Variable(torch.zeros_like(x1.data))
            y2 = torch.autograd.Variable(torch.zeros_like(x1.data))
            x2 = torch.autograd.Variable(torch.zeros_like(x1.data))
            for i in range(x.data.shape[dim]-1,-1,-1):
                y = (a1 * x1 + a2 * x2 + b1 * y1 + b2 * y2)
                x1,x2 = _select_(x,dim,i),x1
                y1, y2 = y, y1
                o.append(y)
            o = torch.cat(o[::-1],dim=dim)
            return o
        x_ = smooth_forward(x,A1,A2,B1,B2,dim=X_DIMENSION)
        x = smooth_backward(x,A3,A4,B1,B2,dim=X_DIMENSION) + x_
        x_ = smooth_forward(x,A1,A2,B1,B2,dim=Y_DIMENSION)
        x = smooth_backward(x,A3,A4,B1,B2,dim=Y_DIMENSION) + x_
        return x
    def gaussian(self, sigma):
        """
            sets the filter density to
            approximate a gaussian filter with 
            sigma standard deviation.
        """
        self.density.data[0] = 1.0/(sigma*variables.default_resolution.pixel_per_degree)


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
            s = convis.filters.SmoothConv(n=6,tau=0.05)
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

class NLRectify(Layer):
    """Rectifies the input (ie. sets values < 0 to 0)
    """
    def __init__(self):
        super(NLRectify, self).__init__()
    def forward(self, inp):
        return (inp).clamp(min=0.0,max=1000000.0)

class NLRectifyScale(Layer):
    """Rectifies the input, but transforms the input with a scale and a bias.

        Pseudocode:

            out = bias + in * scale
            out[out < 0] = 0

    """
    def __init__(self):
        super(NLRectifyScale, self).__init__()
        self.scale = convis.Parameter(1.0)
        self.bias = convis.Parameter(0.0)
    def forward(self, inp):
        return (self.bias+inp*self.scale).clamp(min=0.0,max=1000000.0)

class NLSquare(Layer):
    """A square nonlinearity with a scalable input weight and bias.

    """
    def __init__(self):
        super(NLSquare, self).__init__()
        self.scale = convis.Parameter(1.0)
        self.bias = convis.Parameter(0.0)
    def forward(self, inp):
        return (self.bias+inp*self.scale)**2

class NLRectifySquare(Layer):
    """A square nonlinearity with a scalable input weight and bias
    that cuts off negative values after adding the bias.

    """
    def __init__(self):
        super(NLRectifySquare, self).__init__()
        self.scale = convis.Parameter(1.0)
        self.bias = convis.Parameter(0.0)
    def forward(self, inp):
        return ((self.bias+inp*self.scale).clamp(min=0.0,max=1000000.0))**2
