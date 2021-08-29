import torch
from torch import nn
import numpy as np
import math
from .. import numerical_filters as nf
from .. import variables
from ..base import Layer
from .. import _get_default_resolution
TIME_DIMENSION = 2
X_DIMENSION = 3
Y_DIMENSION = 4

__all__ = ['TimePadding','Delay','VariableDelay','Conv3d','Conv2d','Conv1d','RF','L','LN',
           'TemporalLowPassFilterRecursive','TemporalHighPassFilterRecursive','SpatialRecursiveFilter',
           'SmoothConv','NLRectify','NLSquare','NLRectifyScale','NLRectifySquare',
           'Sum','sum','Diff']

class TimePadding(Layer):
    """
        Remembers references to previous time slices
        and prepends the input with `length` many
        time steps from previous calls.
        
        If the size of the image is changed without
        removing the state first, an Exception is
        raised.

        To avoid this, call :meth:`~convis.base.Layer.clear_state()`. This method is recursive
        on all :class:`convis.base.Layer` s, so you only have to call it on the
        outermost :class:`~convis.base.Layer`.
        If you want to store your history for one set of images,
        do some computation on other images and then return to
        the previous one, you can use :meth:`~convis.base.Layer.push_state()` and :meth:`~convis.base.Layer.pop_state()`.

        Parameters
        ----------
        length : int
            The number of frames that should be prepended to each slice
        mode : str
            The behaviour if the buffer does not contain enough frames:
              - `'mirror'` (default) appends the time reversed input until buffer is filled enough
              - `'full_copy'` appends the input until buffer is full enough
              - `'first_frame'` appends copies of the first frame of the input
              - `'mean'` fills the buffer with the mean value of the input
              - `'ones'` fills the buffer with ones
              - `'zeros'` fills the buffer with zeros
    """
    def __init__(self,length=0,mode='mirror'):
        self.dim = 5
        self.length = length
        super(TimePadding, self).__init__()
        self.register_state('saved_inputs',[])
        self.saved_inputs = []
        self.mode = 'first_frame'
    @property
    def available_length(self):
        return np.sum([i.size()[TIME_DIMENSION] for i in self.saved_inputs])
    def forward(self, x):
        if self.length == 0:
            return x
        if len(self.saved_inputs) > 0:
            if x.size()[-2:] != self.saved_inputs[-1].size()[-2:]:
                raise Exception('input size '+str(x.size()[-2:])+' does not match state size ('+str(self.saved_inputs[-1].size()[-2:])+')! Call `.clear_state()` on your model first!')
        if x.shape[2] > self.length:
            x_offset = x.shape[2] - self.length
        else:
            x_offset = 0
        while self.available_length < self.length:
            if self.mode == 'full_copy':
                self.saved_inputs.append(x[:,:,x_offset:,:,:])
            elif self.mode == 'mirror':
                self.saved_inputs.append(variables.Variable(x.numpy()[:,:,::-1,:,:])[:,:,x_offset:,:,:])
            elif self.mode == 'first_frame':
                self.saved_inputs.append(x[:,:,:1,:,:])
            elif self.mode == 'mean':
                self.saved_inputs.append(torch.ones_like(x[:,:,x_offset:,:,:])*x.mean())
            elif self.mode == 'ones':
                self.saved_inputs.append(torch.ones_like(x[:,:,x_offset:,:,:]))
            elif self.mode == 'zeros':
                self.saved_inputs.append(torch.zeros_like(x[:,:,x_offset:,:,:]))
            else:
                raise Exception("TimePadding argument `mode`='%s' not recognized!."%(self.mode,))
        if self._use_cuda:
            x_pad = torch.cat([i.cuda().detach() for i in self.saved_inputs] + [x.cuda()], dim=TIME_DIMENSION)
        else:
            x_pad = torch.cat([i.cpu().detach() for i in self.saved_inputs] +[x.cpu()], dim=TIME_DIMENSION)
        while self.available_length > self.length - x.shape[2] and len(self.saved_inputs) > 0:
            self.saved_inputs.pop(0)
        self.saved_inputs.append(x[:,:,x_offset:,:,:])
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
        self.delays = variables.Parameter(delays)
        self.all_delay = Delay()
    def forward(self, x):
        if self.delays is None or self.delays.size()[-2:] != x.size()[-2:]:
            self.delays = variables.Parameter(variables.ones(x.size()[-2:]))
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

            * time_pad: True (enables padding in time)
            * autopad: True (enables padding in space)

        To change the weight, use the method `set_weight()`
        which also accepts numpy arguments.


        See Also
        --------
        torch.nn.Conv3d
        Conv1d
        Conv2d
        RF
    """
    def __init__(self,in_channels=1,out_channels=1,kernel_size=(1,1,1),bias=True,*args,**kwargs):
        self.do_adjust_padding = kwargs.get('adjust_padding',False)
        self.do_time_pad = kwargs.get('time_pad',True)
        self.autopad = kwargs.get('autopad',True)
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
        #self.weight.data = torch.zeros(self.weight.data.shape)
        #self.w = self.weight
        self.weight = variables.Parameter(np.zeros(self.weight.data.shape),
                        doc="""The weight tensor of the convolution""")
        if bias is True:
            self.bias = variables.Parameter(np.zeros(self.bias.data.shape),
                            doc="""The bias of the convolution""")
        self.time_pad = TimePadding(self.weight.size()[TIME_DIMENSION])
    def set_weight(self,w,normalize=False,preserve_channels=False,flip=True):
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

            flip: bool (default: True)
                If `True`, the weight will be flipped, so that it corresponds 
                1:1 to patterns it matches (ie. 0,0,0 is the first frame, top left pixel)
                and the impulse response will be exactly `w`.
                If `False`, the weight will not be flipped.

                .. versionadded:: 0.6.4


        """
        if type(w) in [int,float]:
            #self.weight.data = variables.ones(self.weight.data.shape) * w
            self.weight.data = torch.ones(self.weight.data.shape) * w
        else:
            if len(w.shape) == 5:
                self.out_channels = w.shape[0]
                self.in_channels = w.shape[1]
            else:
                if hasattr(w,'__array__'):
                    # convert to numpy if possible
                    w = w.__array__()
                if len(w.shape) == 1:
                    w = w[None,None,:,None,None]
                elif len(w.shape) == 2:
                    w = w[None,None,None,:,:]
                elif len(w.shape) == 3:
                    w = w[None,None,:,:,:]
                if preserve_channels is True:
                    w = w * np.ones((self.out_channels, self.in_channels, 1, 1, 1))
            w = torch.Tensor(w)
            if flip:
                if hasattr(w,'flip'):
                    # in newer PyTorch versions
                    w = w.flip(2,3,4)
                else:
                    # fallback, see https://github.com/pytorch/pytorch/issues/229
                    def flip(x, dim):
                        xsize = x.size()
                        dim = x.dim() + dim if dim < 0 else dim
                        x = x.view(-1, *xsize[dim:])
                        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                                          -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
                        return x.view(xsize)
                    w = flip(w,2)
                    w = flip(w,3)
                    w = flip(w,4)
            self.weight.data = w
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
        return (int(math.floor((k[2])/2.0)),
                int(math.ceil(k[2]))-int(math.floor((k[2])/2.0))-1,
                int(math.floor((k[1])/2.0)),
                int(math.ceil(k[1]))-int(math.floor((k[1])/2.0))-1,
                0,0)
    @property
    def kernel_padding_all(self):
        k = np.array(self.weight.data.shape[2:])
        print('new code!')
        return (int(math.floor((k[2])/2.0))-1,
                int(math.ceil(k[2]))-int(math.floor((k[2])/2.0)),
                int(math.floor((k[1])/2.0))-1,
                int(math.ceil(k[1]))-int(math.floor((k[1])/2.0)),
                int(math.floor((k[0])/2.0))-1,
                int(math.ceil(k[0]))-int(math.floor((k[0])/2)))
    def exponential(self,tau=0.0,adjust_padding=False,*args,**kwargs):
        """Sets the weight to be a 1d temporal lowpass filter with time constant `tau`."""
        self.set_weight(nf.exponential_filter_1d(tau,*args,**kwargs)[::-1].copy(),normalize=False,flip=False)
        if adjust_padding:
            self.adjust_padding()
    def highpass_exponential(self,tau=0.0,adjust_padding=False,*args,**kwargs):
        """Sets the weight to be a 1d temporal highpass filter with time constant `tau`."""
        self.set_weight(nf.exponential_highpass_filter_1d(tau,*args,**kwargs)[::-1].copy(),normalize=False,flip=False)
        if adjust_padding:
            self.adjust_padding()
    def gaussian(self,sig,adjust_padding=False,resolution=None):
        """Sets the weight to be a 2d gaussian filter with width `sig`."""
        self.set_weight(nf.gauss_filter_5d(sig,sig,resolution=resolution),normalize=False)
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

        Simple usage example processing a grating stimulus from `convis.samples`::

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
        torch.nn.Conv3d
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
        if self.rf_placement_mode == 'corner':
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


class Conv2d(nn.Conv2d, Layer):
    """Performs a 2d convolution.

    Filter size can be 2d (spatial filter: `x,y`) or 3d (`channels,x,y`)
    or 4d (`batches,channels,x,y`).

    A filter can be set by supplying a `torch.Tensor` or `np.array` to `.set_weight()` and is expanded to a 4d Tensor.
    **Note:** The filter is flipped during the convolution with respect to the image.


    Convolutions in convis do automatic padding in space, unless told other wise by supplying 
    the keyword argument `autopad=False`. The input will be padded to create output of the same size.
    Uneven weights (eg. `9x9`) will be perfectly centered, such that the center pixel of the weight, the
    input pixel and output pixel all align. For even weights, this holds for the last pixel 
    after the center (`[6,6]` for a `10x10` weight).

    The attribute `self.autopad_mode` can be set to a string that is passed to
    :func:`torch.nn.functional.pad`. The default is `'replicate`'

    See Also
    --------
    torch.nn.Conv2d
    Conv1d
    Conv3d
    RF
    """
    def __init__(self,in_channels=1,out_channels=1,kernel_size=(1,1),*args,**kwargs):
        self.dims = 5
        self.autopad = kwargs.get('autopad',True)
        self.autopad_mode = 'replicate'
        if 'autopad' in kwargs.keys():
            del kwargs['autopad']
        super(Conv2d, self).__init__(in_channels,out_channels,kernel_size,*args,**kwargs)
        if hasattr(self,'bias') and self.bias is not None:
            self.bias = variables.Parameter(0.0)
        self.weight = variables.Parameter(torch.zeros(self.weight.data.shape))
    def set_weight(self,w,normalize=False,flip=True):
        """
            Sets a new weight for the convolution.

            Parameters
            ----------
            w: numpy array or PyTorch Tensor
                The new kernel `w` should have 2,3 or 4 dimensions:
                **2d:** (x, y)
                **3d:** (out_channels, x, y)
                **4d:** (in_channels, out_channels, x, y)
                Missing dimensions are added at the front.

            normalize: bool (default: False)
                Whether or not the sum of the kernel values
                should be normalized to 1, such that the
                sum over all input values and all output 
                values is the approximately same.

            flip: bool (default: True)
                If `True`, the weight will be flipped, so that it corresponds 
                1:1 to patterns it matches (ie. 0,0 is the top left pixel)
                and the impulse response will be exactly `w`.
                If `False`, the weight will not be flipped.

                .. versionadded:: 0.6.4

        """
        if type(w) in [int,float]:
            self.weight.data = torch.ones(self.weight.data.shape).data * w
        else:
            if self.weight.data.shape == w.shape:
                self.weight.data = torch.Tensor(w)
            else:
                if len(w.shape) == 4:
                    self.out_channels = w.shape[0]
                    self.in_channels = w.shape[1]
                    w_h = w.shape[2]
                    w_w = w.shape[3]
                    w = torch.Tensor(w)
                elif len(w.shape) == 3:
                    w_h = w.shape[1]
                    w_w = w.shape[2]
                    w = torch.Tensor(w)[None]
                elif len(w.shape) == 2:
                    w_h = w.shape[0]
                    w_w = w.shape[1]
                    w= torch.Tensor(w)[None][None]
                else:
                    raise Exception('Conv2d accepts weights with 2,3 or 4 dimensions. Weight has shape '+str(w.shape)+'!')
            if flip:
                if hasattr(w,'flip'):
                    # in newer PyTorch versions
                    w = w.flip(2,3)
                else:
                    # fallback, see https://github.com/pytorch/pytorch/issues/229
                    def flip(x, dim):
                        xsize = x.size()
                        dim = x.dim() + dim if dim < 0 else dim
                        x = x.view(-1, *xsize[dim:])
                        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                                          -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
                        return x.view(xsize)
                    w = flip(w,2)
                    w = flip(w,3)
            self.weight.data = w
        if normalize:
            self.weight.data = self.weight.data / self.weight.data.sum()
    @property
    def kernel_padding(self):
        k = np.array(self.weight.data.shape[-2:])
        return (int(math.floor((k[1])/2.0)),
                int(math.ceil(k[1]))-int(math.floor((k[1])/2.0))-1,
                int(math.floor((k[0])/2.0)),
                int(math.ceil(k[0]))-int(math.floor((k[0])/2.0))-1,
                0,0)
    def forward(self,x):
        if self.autopad:
            x = torch.nn.functional.pad(x,self.kernel_padding, self.autopad_mode)
        outs = []
        for i in range(x.size()[0]):
            outs.append(super(Conv2d, self).forward(x[i].transpose(0,1)).transpose(0,1)[None,:,:,:,:])
        return torch.cat(outs,dim=0)
    def gaussian(self,sig):
        self.set_weight(nf.gauss_filter_2d(sig,sig)[None,None,:,:],normalize=False)


class Conv1d(nn.Conv1d, Layer):
    """1d convolution with optional in-out-channels.

    Weights can be set with `set_weight` and will be automatically flipped 
    to keep the weight and the impulse response identical.

    The weight can be 1d (no channels/only time) or 3d (in-channels, out-channels,time).

    .. note::
        
        During the processing, all spatial dimensions will be collapsed into the batch dimension.


    See Also
    --------
    torch.nn.Conv1d
    Conv2d
    Conv3d
    RF
    """
    def __init__(self,in_channels=1,out_channels=1,kernel_size=1,*args,**kwargs):
        self.do_time_pad = kwargs.get('time_pad',True)
        if 'time_pad' in kwargs.keys():
            del kwargs['time_pad']
        super(Conv1d, self).__init__(in_channels,out_channels,kernel_size,*args,**kwargs)
        if hasattr(self,'bias') and self.bias is not None:
            self.bias.data[0] = 0.0
        self.weight.data = torch.zeros(self.weight.data.shape)
        self.time_pad = TimePadding(self.weight.size()[TIME_DIMENSION])
    @property
    def filter_length(self):
        return self.weight.data.shape[-1] - 1
    def set_weight(self,w,normalize=False,flip=True):
        """
            Sets a new weight for the convolution.

            Parameters
            ----------
            w: numpy array or PyTorch Tensor
                The new kernel `w` should have 1 or 3 dimensions:
                **1d:** (time)
                **3d:** (in_channels, out_channels, time)

            normalize: bool (default: False)
                Whether or not the sum of the kernel values
                should be normalized to 1, such that the
                sum over all input values and all output 
                values is the approximately same.

            flip: bool (default: True)
                If `True`, the weight will be flipped, so that it corresponds 
                1:1 to patterns it matches (ie. 0 is the first frame)
                and the impulse response will be exactly `w`.
                If `False`, the weight will not be flipped.

                .. versionadded:: 0.6.4

        """
        if type(w) in [int,float]:
            self.weight.data = torch.ones(self.weight.data.shape).data * w
        else:
            #self.weight.data = torch.Tensor(w)
            w = torch.Tensor(w)
            if len(w.shape) == 3:
                self.out_channels = w.shape[0]
                self.in_channels = w.shape[1]
            elif len(w.shape) == 1:
                w = w[None,None,:]
            else:
                raise Exception('Conv1d weights have to be 1d or 3d, not '+str(len(w.shape))+'! Please refer to the doc string.')
            if flip:
                if hasattr(w,'flip'):
                    # in newer PyTorch versions
                    w = w.flip(2)
                else:
                    # fallback, see https://github.com/pytorch/pytorch/issues/229
                    def flip(x, dim):
                        xsize = x.size()
                        dim = x.dim() + dim if dim < 0 else dim
                        x = x.view(-1, *xsize[dim:])
                        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                                          -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
                        return x.view(xsize)
                    w = flip(w,2)
            self.weight.data = w
        if normalize:
            self.weight.data = self.weight.data / self.weight.data.sum()
    def forward(self,x):
        if self.do_time_pad:
            self.time_pad.length = self.filter_length
            x = self.time_pad(x)
        # we move both space dimensions into the batch dimension
        s = list(x.size())
        x = x.transpose(1,3).transpose(2,4).reshape((s[0]*s[3]*s[4],s[1],s[2]))
        y = super(Conv1d, self).forward(x)
        s_y = list(y.size())
        return y.reshape((s[0],s[3],s[4],s[1],s_y[2])).transpose(4,2).transpose(3,1)
    def exponential(self,tau,*args,**kwargs):
        """Sets the weight to be an exponential filter (low-pass filter) with time constant `tau`.
        """
        self.set_weight(nf.exponential_filter_1d(tau,*args,**kwargs),normalize=False,flip=True)

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
        self.tau = variables.Parameter(torch.Tensor([0.01]),requires_grad=requires_grad)
        self.register_state('last_y',None)
    def clear(self):
        if hasattr(self,'last_y'):
            self.last_y = None
    def forward(self, x):
        steps = variables.Parameter(1.0/_get_default_resolution().steps_per_second,requires_grad=False)
        if self._use_cuda:
            steps = steps.cuda()
        a_0 = 1.0
        a_1 = -torch.exp(-steps/self.tau)
        b_0 = 1.0 - a_1
        if self.last_y is not None:
            y = self.last_y
        else:
            y = variables.zeros(1,1,1,x.data.shape[3],x.data.shape[4])
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
        self.tau = variables.Parameter(torch.Tensor([0.01]),requires_grad=requires_grad)
        self.k = variables.Parameter(torch.Tensor([0.5]),requires_grad=requires_grad)
        self.register_state('last_y',None)
    def clear(self):
        if hasattr(self,'last_y'):
            self.last_y = None
    def forward(self, x):
        steps = variables.Parameter(1.0/_get_default_resolution().steps_per_second,requires_grad=False)
        if self._use_cuda:
            steps = steps.cuda()
        a_0 = 1.0
        a_1 = -torch.exp(-steps/self.tau)
        b_0 = 1.0 - a_1
        if self.last_y is not None:
            y = self.last_y
        else:
            y = variables.zeros(1,1,1,x.data.shape[3],x.data.shape[4])
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
        self.density = variables.Parameter(torch.Tensor([1.0]))
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
            y1 = variables.zeros_like(x1.data)
            y2 = variables.zeros_like(x1.data)
            x2 = variables.zeros_like(x1.data)
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
            y1 = variables.zeros_like(x1.data)
            y2 = variables.zeros_like(x1.data)
            x2 = variables.zeros_like(x1.data)
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
        self.density.data[0] = 1.0/(sigma*_get_default_resolution().pixel_per_degree)


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

    Note
    ----

    To implement simple nonlinearities, you can also
    use lambda expressions::

        model = convis.mdoels.LN()
        model.nonlinearity = lambda inp: (inp).clamp(min=0.0,max=1000000.0)
    """
    def __init__(self):
        super(NLRectify, self).__init__()
    def forward(self, inp):
        return (inp).clamp(min=0.0,max=1000000.0)

class NLRectifyScale(Layer):
    """Rectifies the input, but transforms the input with a scale and a bias.

        Pseudocode::

            out = bias + the_input * scale
            out[out < 0] = 0

    """
    def __init__(self):
        super(NLRectifyScale, self).__init__()
        self.scale = variables.Parameter(1.0)
        self.bias = variables.Parameter(0.0)
    def forward(self, inp):
        return (self.bias+inp*self.scale).clamp(min=0.0,max=1000000.0)

class NLSquare(Layer):
    """A square nonlinearity with a scalable input weight and bias.

    """
    def __init__(self):
        super(NLSquare, self).__init__()
        self.scale = variables.Parameter(1.0)
        self.bias = variables.Parameter(0.0)
    def forward(self, inp):
        return (self.bias+inp*self.scale)**2

class NLRectifySquare(Layer):
    """A square nonlinearity with a scalable input weight and bias
    that cuts off negative values after adding the bias.

    """
    def __init__(self):
        super(NLRectifySquare, self).__init__()
        self.scale = variables.Parameter(1.0)
        self.bias = variables.Parameter(0.0)
    def forward(self, inp):
        return ((self.bias+inp*self.scale).clamp(min=0.0,max=1000000.0))**2

def sum(*args, **kwargs):
    """concatenates and sums tensors over a given dimension `dim`.

    Examples
    --------

        >>> inp = convis.prepare_input(np.ones((2,2,100,10,10)))
        >>> o = convis.filters.sum(inp,inp,inp,dim=1)

    See Also
    --------
    Sum
    """
    dim = kwargs.get('dim',0)
    return torch.cat(args,dim=dim).sum(dim=dim,keepdim=True)

class Sum(Layer):
    """A Layer that combines all inputs into one tensor and 
    sums over a given dimension.
    
    Can be used to collapse batch or filter dimensions.

    Examples
    --------

        >>> s = Sum(1)
        >>> inp = convis.prepare_input(np.ones((2,2,100,10,10)))
        >>> o = s(inp,inp,inp)
    
    See Also
    --------
    sum
    """
    def __init__(self, dim=0):
        self.dims = 5
        self.dim = dim
        super(Sum, self).__init__()
    def forward(self, *args):
        return sum(args,dim=self.dim)


class Diff(Layer):
    """Takes the difference between two consecutive frames.


    Example
    -------

    .. plot::
        :include-source:
    
        import convis
        d = Diff()
        inp = convis.samples.moving_grating()
        o = d.run(inp,dt=200)
        o.plot()


    """
    def __init__(self):
        self.dim = 5
        super(DVS2,self).__init__()
        self.register_state('last_frame',None)
    def forward(self,inp):
        if self.last_frame is not None:
            first_frame = inp[:,:,:1,:,:] - self.last_frame
        else:
            first_frame = torch.zeros((inp.size()[0],inp.size()[1],1,inp.size()[3],inp.size()[4]))
        self.last_frame = inp[:,:,-1:,:,:]
        return torch.cat([first_frame, inp[:,:,1:,:,:] - inp[:,:,:-1,:,:]],dim=2)


from . import simple
from . import retina
from . import spiking