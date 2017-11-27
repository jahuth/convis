.. _filters:
Filters and Layers
=====================

Convolution Filters
-------------------------

`covnis.filters.Conv3d` is the basic 3d convolution filter and based on `torch.nn.Conv3d`.
It can dynamically pad the input in `x` and `y` dimensions to keep the output the same size as the input.
The time dimension has to be padded by another operation.

.. code::
    conv = covnis.filters.Conv3d(in_channels=1,out_channels=1,kernel_size=(10,20,20),bias=False,)
    print(conv.weight) # the weight parameter
    print(conv.bias)   # the bias parameter
    conv.set_weight(np.random.rand(10,20,20),normalize=True)
    some_output = conv(some_input)



.. warning::
    Currently, the nd convolution operation is leaking memory!

.. autoclass:: convis.filters.Conv3d
   :members:

Spatial gaussian filters and temporal exponential filters can be generated with functions from the `convis.numerical_filters` submodule.

.. plot::

    import convis
    import numpy as np
    import matplotlib.pylab as plt
    plt.figure()
    plt.imshow(convis.numerical_filters.gauss_filter_2d(4.0,4.0))
    plt.show()


.. plot::

    import convis
    import numpy as np
    import matplotlib.pylab as plt
    plt.figure()
    plt.plot(convis.numerical_filters.exponential_filter_1d(tau=0.01))
    plt.show()

Recursive Temporal Filters
-------------------------

There are two filters available that perform recursive temporal filtering.
The advantage over convolutional filtering is that it uses a lot less memory and is a lot faster to compute.
However, the temporal filter is also more simplified and might not be able to fit a specific temporal profile well.

`TemporalLowPassFilterRecursive` is an exponential filter that cuts off high frequencies, while `TemporalHighPassFilterRecursive` is the inverse (a 1, followed by a negative exponential) and cuts off low frequencies.

.. plot::
    import matplotlib.pyplot as plt
    import numpy as np
    import convis
    # random amplitudes and phases for a range of frequencies
    signal = np.sum([np.random.randn()*np.sin(np.linspace(0,2.0,5000)*(freq) 
                     + np.random.rand()*2.0*np.pi) 
                     for freq in np.logspace(-2,7,136)],0)
    f1 = convis.filters.simple.TemporalLowPassFilterRecursive()
    f1.tau.data[0] = 0.005
    f2 = convis.filters.simple.TemporalHighPassFilterRecursive()
    f2.tau.data[0] = 0.005
    f2.k.data[0] = 1.0
    plt.plot(signal)
    plt.plot(f2(signal[None,None,:,None,None]).data.numpy().mean((0,1,3,4)).flatten())
    plt.plot(f1(signal[None,None,:,None,None]).data.numpy().mean((0,1,3,4)).flatten())
    signal_f = np.fft.fft(signal)
    o1 = f1(signal[None,None,:,None,None]).data.numpy().mean((0,1,3,4)).flatten()
    o2 = f2(signal[None,None,:,None,None]).data.numpy().mean((0,1,3,4)).flatten()
    o1_f = np.fft.fft(o1)
    o2_f = np.fft.fft(o2)
    plt.figure()
    plt.plot(np.abs(o1_f)[:2500]/np.abs(signal_f)[:2500])
    plt.plot(np.abs(o2_f)[:2500]/np.abs(signal_f)[:2500])
    plt.xlabel('Frequency')
    plt.ylabel('Gain')
    plt.title('Transfer Functions of Filters')
    plt.gca().set_xscale('log')
    plt.legend(['Low Pass Filter', 'High Pass Filter'])

.. automodule:: convis.filters.simple
   :members:


Retina Layers
---------------

.. automodule:: convis.filters.retina
   :members:


