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


Configuring the Retina Model directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The best way to configure the model is by exploring the 
structure with tab completion of the `.p.` parameter list.
The retina model will give you first the list of layers and
then the list of parameters of each layer.

To change the values, you can use the method `.set`, or 
(*but only if you use the `.p.` list*) by assigning a new value
to the parameter directly.

    >>> retina = convis.retina.Retina()
    >>> retina.p.<tab>
        opl, bipolar, gang_0_input, gang_0_spikes, gang_1_input, gang_1_spikes
    >>> retina.p.bipolar.lambda_amp
    Parameter containing:
    tensor([ 0.])
    >>> retina.p.bipolar.lambda_amp.set(100.0)
    >>> retina.p.bipolar.lambda_amp = 100.0
    >>> retina.p.bipolar.lambda_amp
    Parameter containing:
    tensor([ 100.])

.. note::

    What will not work: 

        >>> retina.bipolar.lambda_amp = 100.0      # <- .p is missing!
        >>> retina.p.bipolar["lambda_amp"] = 100.0 # 

    In both cases, the Parameter will be **replaced** by the number `100.0`. It will no longer be 
    Instead you can use `.set()`: 

        >>> retina.bipolar.lambda_amp.set(100.0)
        >>> retina.p.bipolar["lambda_amp"].set(100.0)
        >>> retina.p.bipolar["lambda_amp"] = Parameter(100.0, doc="new parameter replacing the old one")   

Also you can get a dictionary of configuration values and change the parameters there or save and
load them to and from a json file:

    >>> d = retina.get_parameters()
    >>> d['opl_opl_filter_surround_E_tau']
    array([0.004], dtype=float32)
    >>> d['opl_opl_filter_surround_E_tau'][0] = 0.001
    >>> retina.set_parameters(d)


Configurating the Retina Model with xml files
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