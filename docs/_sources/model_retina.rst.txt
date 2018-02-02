.. _model_retina:

Retina Model
------------------------

This module implements a spiking retina model in python and theano.

It is based on the VirtualRetina Simualtor [Wohrer 2008].


General Overview
~~~~~~~~~~~~~~~~

The formulas on which the classes are based are:

$$C(x,y,t) = G * T(wu,Tu) * E(n,t) * L (x,y,t)$$
$$S(x,y,t) = G * E * C(x,y,t)$$ 
$$I_{OLP}(x,y,t) = \lambda_{OPL}(C(x,y,t) - w_{OPL} S(x,y,t)_)$$ 
$$\\\\frac{dV_{Bip}}{dt} (x,y,t) = I_{OLP}(x,y,t) - g_{A}(x,y,t)dV_{Bip}(x,y,t)$$
$$g_{A}(x,y,t) = G * E * Q(V{Bip})(x,y,t)`with $Q(V{Bip}) = g_{A}^{0} + \lambda_{A}V^2_{Bip}$$
$$I_{Gang}(x,y,t) = G * N(eT * V_{Bip})$$

with :math:`N(V) = \\\\frac{i^0_G}{1-\lambda(V-v^0_G)/i^0_G}` (if :math:`V < v^0_G`)

with :math:`N(V) = i^0_G + \lambda(V-v^0_G)` (if  :math:`V > v^0_G`)


.. image:: _static/dot_test_3.png

The Retina Model class and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The retina model is :mod:`convis.retina.Retina` and can be configured
with a :mod:`convis.retina.RetinaConfiguration` that can be loaded
from a VirtualRetina xml file:

    >>> retina = convis.retina.Retina()
    >>> conf = convis.retina.RetinaConfiguration()
    >>> conf.load_xml('some_file.xml')
    >>> conf.set(.., ..) # changing some values before configuring the model
    >>> retina.parse_config(conf)
    >>> # or more simply:
    >>> retina.parse_config('some_file.xml')

Retina Filters
~~~~~~~~~~~~~~~~

The stages of the VirtualRetina model correspond to the :mod:`convis` classes:

     * :class:`convis.filters.retina.OPL` holds only a reference to the actually used opl implementation:
        - :class:`convis.filters.retina.RecursiveOPLFilter` (all recursive)
        - :class:`convis.filters.retina.HalfRecursiveOPLFilter` (default, temporally recursive, spatial convolution)
        - :class:`convis.filters.retina.SeperatableOPLFilter` (spatial and temporal convolutions, but still computed as separate filters)
        - :class:`convis.filters.retina.FullConvolutionOPLFilter` (a single spatio-temporal convolution filter)
     * :class:`convis.filters.retina.Bipolar` performs contrast gain control
     * :class:`convis.filters.retina.GanglionInput` performs spatial pooling and a static non-linearity
     * :class:`convis.filters.retina.GanglionSpikes` creates spikes