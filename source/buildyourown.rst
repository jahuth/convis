.. _build-your-own:

Build your own Model
====================

To run the examples, we assume that you ran a short preamble before
to include modules and generate some sample input::

.. code-block::
    python

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

Note that the non-linearity is not defined as its own layer here, but as a manipulation of the output of the previous layer (`rf.graph`).

.. code-block::
    python

    model = convis.models.LN()
    model.conv.set_weight(convis.samples.gabor_kernel(phi=0.0)[None,:,:])

Then the model can be executed like this:

.. code-block::
    python

    o = model.run(the_input)

The plot shows that the model responds strongly to some orientations in the stimulus, but not to others:

.. code-block::
    python
    
    plt.plot(o[0][:,10,:],alpha=0.5)


Defining a new Layer
------------------------

