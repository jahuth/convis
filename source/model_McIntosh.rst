
A Convolutional Retina Model
==============================

Lane McIntosh et al. published a paper in 2016 about a machine learning
approach to retinal modelling [1]. They used a convolutional neural network
to fit the responses of retinal ganglion cells.

The model is implemented in convis, but if you want to know how it works, 
you can follow this build-it-yourself recipie.

First, let's start with importing the necessary libraries.

.. code-block:: python

    %matplotlib inline
    import numpy as np
    import matplotlib.pylab as plt
    import convis
    import torch

The model by McIntosh et al. takes an input video and convolves it
in two stages and finally takes a dense linear filter to get a single
time course for each cell.

3d convolutions in PyTorch follow the convention of having
 
 * input dimensions: batch, input channels, time, space x, space y
 * output dimensions: batch, output channels, time, space x, space y
 * weight dimensions: input channels, output channels, time, space x, space y

For us, the batches are always 1 and for most models, the number of
input and output channels are 1 as well. However, since this is an
actual convolutional network, there will be more than one input and
output channel for each layer.

The first layer takes gray scale input and convolves it with 8 subunit
filters, resulting in a :py:`1, 8, time, space x, space y` output.

The second layer has 16 subunits, so it takes the 8 channels from the previous
layer and creates output with 16 channels from it.

To make apparent how :py:module:`convis` and :py:module:`PyTorch` differ,
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
and remembering a state. A similar one is already implemented in convis under :py:module:`convis.filters.`.

So the computation that we want to do is the following:

.. code-block:: python

    # assuming we recieved some `the_input` variable
    activity = convolve_1(the_input)
    activity = relu(activity)
    activity = convolve_2(activity)
    activity = relu(activity)
    activity = linear_readout(activity)

Each of the convolution operations is not a stateless function,
but a convolutional layer that keeps track of its weights and states.
The `relu` (or some other activation function) can be found in the 
:py:module:`torch.nn.function` submodule.
And the readout is a Linear layer that combines the channels and
space together.

To create a model, we define a class that inherits from `convis.Layer`.
In its `__init__` function it has to create all the layers and parameters
that it's using and in its `forward` method, it just does exactly the
computation we outlined in pseudo code before.

.. code-block:: python

    class MyMcIntoshModel(convis.Layer):
        def __init__(self,filter_size=(10,5,5), random_init=True, out_channels=1, filter_2_size=(1,1,1)):
            super(MyMcIntoshModel,self).__init__()
            c1 = MemoryConv(1,8,filter_size)
            self.add_module('c1',c1)
            self.c1.conv.set_weight(1.0,normalize=True)
            if random_init:
                self.c1.conv.set_weight(rand(8,1,filter_size[0],filter_size[1],filter_size[2]),normalize=True)
            c2 = MemoryConv(8,16,filter_2_size)
            self.add_module('c2',c2)
            self.c2.conv.set_weight(1.0,normalize=True)
            if random_init:
                self.c2.conv.set_weight(rand(16,8,filter_2_size[0],filter_2_size[1],filter_2_size[2]),normalize=True)
            self.readout = convis.base.torch.nn.Linear(16,out_channels,bias=False)
        def forward(self, the_input):
            a = convis.base.torch.nn.functional.relu(self.c1(the_input))
            a = convis.base.torch.nn.functional.relu(self.c2(a))
            # The readout should consider all channels and all locations
            # so we need to reshape the Tensor such that the 4th dimension
            # contains dimensions 1,3 and 4
            #  - moving dimension 3 to 4:
            a = torch.cat(a.split(1,dim=3),dim=4)
            #  - moving dimension 1 to 4:
            a = torch.cat(a.split(1,dim=1),dim=4)
            if m.readout.weight.size()[-1] != a.size()[-1]:
                print 'Resetting weight'
                if self._use_cuda:
                    m.readout.weight = torch.nn.Parameter(torch.ones((m.readout.weight.size()[0],a.size()[-1])))
                    m.readout.cuda()
                else:
                    m.readout.weight = torch.nn.Parameter(torch.ones((m.readout.weight.size()[0],a.size()[-1])))
            a = self.readout(a)
            return a

Now that the model is defined, we can immediately use it and fit it to data.

.. code-block:: python

    m = MyMcIntoshModel(filter_size=(10,10,10))
    inp = convis.samples.moving_grating()*convis.samples.chirp()
    inp = convis.prepare_input(inp,cuda=True)
    o = m.run(inp, dt = 100)
    o.plot()

    m.set_optimizer.LBFGS([m.readout.weight])
    opt = m.optimize(inp,inp[:,:,:,:1,:1],dt=100)

    plt.figure(figsize=(10,6))
    o = m.run(inp, dt = 100)
    plt.plot(inp.data.cpu().numpy()[0,0,:,0,0], lw=2)
    convis.plot_5d_time(o[0], mean = (0,1,3), lw=2)
    plt.xlabel('time')
    plt.ylabel('response')
    plt.legend(['target','model'])




[1] McIntosh, L. T., Maheswaranathan, N., Nayebi, A., Ganguli, S., & Baccus, S. A. (2016). Deep Learning Models of the Retinal Response to Natural Scenes. Advances in Neural Information Processing Systems 29 (NIPS), (Nips), 1–9.