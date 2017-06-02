.. _build-your-own:

Build your own Model
====================

To run the examples, we assume that you ran a short preamble before
to include modules and generate some sample input::

    %matplotlib inline

    import numpy as np
    import matplotlib.pylab as plt

    import convis
    v = 10.0
    the_input = np.concatenate([contrast * convis.samples.moving_bars(t=200, vt=20, vx=v*np.sin(phi), vy=v*np.cos(phi)) 
                                for phi in np.linspace(0,360.0,30.0) for contrast in [0.0,1.0]], axis = 0)


An orientation selective LN model
---------------------------------

This example creates a visual model with a two-dimensional receptive field which has the shape of a gabor patch.
To discard anti-phase responses which would anihilate the mean response to the stimulus, the output is half-wave rectified and also squared to amphasize strong responses.

Note that the non-linearity is not defined as its own layer here, but as a manipulation of the output of the previous layer (`rf.graph`). ::

    rf = convis.filters.simple.ConvolutionFilter2d(
        {'kernel': convis.samples.gabor_kernel(phi=0.0) },
        name='ReceptiveField')

    nonlinearity = lambda x: x.clip(0,1000)**2 
    # theano tensors support many operations of numpy arrays

    m = convis.make_model(nonlinearity(rf.graph))

Then the model can be executed like this::

    o = m.run(the_input)

The plot shows that the model responds strongly to some orientations in the stimulus, but not to others::

    plt.plot(o[0][:,10,:],alpha=0.5)


Defining a new Layer
------------------------

To define a new layer, a few points have to be observed:

 1. the layer should inherit from the `convis.base.N` (or `convis.N`) class
 2. the layer should contain a `default_input` which is a sum of 3d or 5d tensors, so that inputs can be added to the layer
     * you can create such an input by invoking `self.create_input()` in your constructor
     * or call `self.create_input("some_name")` to create named inputs if you want to expose more than one
 3. supply the `graph` ie. the overall output of your layer to the super constructor:
     * if you have more than one output, put them in a list
     * Replace 'NameOfYourClass' with the name of your class in the expression: `super(NameOfYourClass, self).__init__(graph, ...)`
 4. (optional) think about which configuration options you want to expose
     * if you do not need any configuration, set `self.expects_config` to False
     * if you want to initialize a parameter from the configuration, follow the example of 'b' in the code

Example ::

    rf = convis.filters.simple.ConvolutionFilter2d(
        {'kernel': convis.samples.gabor_kernel(phi=0.0) },
        name='ReceptiveField')

    class Nonlinearity(convis.N):
        def __init__(self,name='NonLinearity',**kwargs):
            a = convis.shared_parameter(0.0, name='a')
            b = convis.shared_parameter(1.0, name='b', config_key='b')
            graph = a + b * self.create_input().clip(0,1000)
            super(Nonlinearity, self).__init__(graph,name=name,**kwargs)

    nonlinearity = Nonlinearity(config={'b':2.0})
    temporalfilter = convis.ConvolutionFilter1d(
        {'kernel': convis.numerical_filters.exponential_filter_1d(tau=0.01)}, 
        name='E')
    convis.connect([rf,temporalfilter, nonlinearity])
    m = convis.make_model(nonlinearity)
    o = m.run(the_input)
    plt.plot(o[0][:1000,10,:],alpha=0.5)
    plt.xlabel('time (eg. ms)')


Subunit models
--------------

In this model, convolutional filters are combined with a receptive
field filter to create a single subunit model. ::

    receptors = convis.filters.simple.ConvolutionFilter2d(
        {'kernel': convis.numerical_filters.gauss_filter_2d(2.0,2.0) },
        name='ReceptorLayer')
    horizontal_cells = convis.filters.simple.ConvolutionFilter2d(
        {'kernel': convis.numerical_filters.gauss_filter_2d(4.0,4.0) },
        name='HorizontalLayer')
    rf = convis.RF_2d_kernel_filter(
        {'kernel': convis.samples.gabor_kernel(size=the_input.shape[1]) },
        name='ReceptiveField')


    horizontal_cells += receptors
    rf += receptors
    rf += -0.5*horizontal_cells.graph

    m = convis.make_model(rf)
    o = m.run(the_input)
    plt.plot(o[0][:1000,10,:],alpha=0.5)
    plt.xlabel('time (eg. ms)')

    plt.plot(o[0][:,:,:].clip(0,None).mean((1,2)),alpha=0.5)

.. plot::

    import numpy as np
    import matplotlib.pylab as plt

    import convis
    v = 10.0
    the_input = np.concatenate([contrast * convis.samples.moving_bars(t=200, vt=20, vx=v*np.sin(phi), vy=v*np.cos(phi)) 
                                for phi in np.linspace(0,360.0,30.0) for contrast in [0.0,1.0]], axis = 0)

    receptors = convis.filters.simple.ConvolutionFilter2d(
        {'kernel': convis.numerical_filters.gauss_filter_2d(2.0,2.0) },
        name='ReceptorLayer')
    horizontal_cells = convis.filters.simple.ConvolutionFilter2d(
        {'kernel': convis.numerical_filters.gauss_filter_2d(4.0,4.0) },
        name='HorizontalLayer')
    rf = convis.RF_2d_kernel_filter(
        {'kernel': convis.samples.gabor_kernel(size=the_input.shape[1]) },
        name='ReceptiveField')


    horizontal_cells += receptors
    rf += receptors
    rf += -0.5*horizontal_cells.graph

    m = convis.make_model(rf)
    o = m.run(the_input)
    plt.plot(o[0][:1000,10,:],alpha=0.5)
    plt.xlabel('time (eg. ms)')

    plt.plot(o[0][:,:,:].clip(0,None).mean((1,2)),alpha=0.5)
    plt.show()


The structure of the model::

    input -> ReceptorLayer -> HorizontalLayer -> output
                           -> ReceptiveField -> output

