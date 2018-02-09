.. _pytorch:
PyTorch Basics
===============

PyTorch is a computing library with a focus on deep learning.
It provides three mayor submodules to make deep learning easy:

 * A high performance tensor computing package `torch.tensor`
 * A computational graph that is built while you do your computations `torch.autograd`
 * Classes to package computations into modules and collect parameters hierarchically `torch.nn`


Tensor computing with `torch.tensor`
----------------------------------------

Tensors can be created from numpy arrays:

.. code::
    python

    >>> a = torch.from_numpy(np.random.randn(10,10,10,10))
    >>> b = torch.DoubleTensor(np.random.randn(10,10,10,10))
    >>> (a + b).mean()
    -0.006650402067610719



Automated differentiation with `torch.autograd`
-------------------------------------------------

.. code::
    python

    >>> a = torch.autograd.Variable(torch.from_numpy(np.random.randn(10,10,10,10)),requires_grad=True)
    >>> b = torch.autograd.Variable(torch.from_numpy(np.random.randn(10,10,10,10)),requires_grad=True)
    >>> c = (a + b).mean()
    >>> c.backward(retain_graph=True) # adds the backpropagated gradient to the gradient buffers
    >>> a.grad
    # tensor of 1s
    >>> b.grad
    # tensor of 1s


Model building with `torch.nn`
------------------------------


PyTorch Extensions in Convis
===============================



Layer
---------

Layers are extensions of `torch.nn.Module`s. They behave very similarly, but have a few additional features:

 * a Layer knows if it accepts 1d, 3d, or 5d time sequence input and can broadcast the input accordingly if it has too few dimensions
 * instead of running the model on the complete time series, the input can be automatically chunked by using the `.run(.., dt=chunk_length)` method instead of calling the Layer directly.
 * a Layer can create its own optimizer



Output
---------

A class that collects all outputs of a Layer for Layers that have more than one output.



Extending Conv3d
------------------
.. _filters_Conv3d:

To make apparent how :py:mod:`convis` and :py:mod:`PyTorch <torch>` differ,
we will first implement a custom convolution layer that wraps the `PyTorch`
3d convolution.

To create an output that is the same shape as the input, we need to pad
the input at both sides of the x and y dimension, with either a constant,
a mirror or a replicating border condition, and we need to remember the
the last slice of the previous input, so that we can continously take in
input and not lose frames between them.

So what we want the layer to do in its forward pass is:

.. code-block:: python

        def forward(self, x):
            if not ... :
                # for the case that we have no input input_state
                # or the input state does not match the shape of x
                self.input_state = torch.autograd.Variable(torch.zeros(...))
                # using eg. the first slice of the input initially
                self.input_state[:,:,-self.filter_length:,:,:] = x[:,:,:self.filter_length,:,:]
            x_pad = torch.cat([self.input_state, x], dim=2) # input padded in time
            self.input_state = x_pad[:,:,-(self.filter_length):,:,:]
            # finally, padding x and y dimension
            x = torch.nn.functional.pad(x,self.kernel_padding, 'replicate')
            return self.conv(x_pad)

A full implementation can look something like this:

.. code-block:: python

    class MyMemoryConv(convis.Layer):
        def __init__(self,in_channels=1,out_channels=1,kernel_dim=(1,1,1), bias = False):
            self.dim = 5
            self.autopad = True
            super(MyMemoryConv, self).__init__()
            self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_dim, bias = bias)
            self.input_state = None
        @property
        def filter_length(self):
            """The length of the filter in time"""
            return self.conv.weight.data.shape[2] - 1
        @property
        def kernel_padding(self):
            """The x and y dimension padding"""
            k = np.array(self.weight.data.shape[2:])
            return (int(math.floor((k[2])/2.0))-1,
                    int(math.ceil(k[2]))-int(math.floor((k[2])/2.0)),
                    int(math.floor((k[1])/2.0))-1,
                    int(math.ceil(k[1]))-int(math.floor((k[1])/2.0)),
                    0,0)
        def set_weight(self,w,normalize=False):
            if type(w) in [int,float]:
                self.conv.weight.data = torch.ones(self.conv.weight.data.shape) * w
            else:
                if len(w.shape) == 1:
                    w = w[None,None,:,None,None]
                if len(w.shape) == 2:
                    w = w[None,None,None,:,:]
                if len(w.shape) == 3:
                    w = w[None,None,:,:,:]
                self.conv.weight.data = torch.Tensor(w)
                self.conv.kernel_size = self.conv.weight.data.shape[2:]
            if normalize:
                self.conv.weight.data = self.conv.weight.data / self.conv.weight.data.sum()
        def forward(self, x):
            if (self.input_state is None or 
                   self.input_state.size()[:2] != x.size()[:2] or 
                   self.input_state.size()[-2:] != x.size()[-2:]):
                self.input_state = x.detach()
            if self.filter_length > 0:
                if self._use_cuda:
                    x_pad = torch.cat([self.input_state[:,:,-(self.filter_length):,:,:].cuda(), x.cuda()], dim=TIME_DIMENSION)
                    self.conv.cuda()
                else:
                    x_pad = torch.cat([self.input_state[:,:,-(self.filter_length):,:,:], x], dim=TIME_DIMENSION)
            else:
                x_pad = x
            self.input_state = x.detach()
            x_pad = torch.nn.functional.pad(x_pad,self.kernel_padding, 'replicate')
            return self.conv(x_pad)


Now this convolution layer already does most of the hard work of padding the input
and remembering a state. A similar one is already implemented in convis under :py:mod:`convis.filters`.
