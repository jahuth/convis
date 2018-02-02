
VirtualRetina-like Simulator
=============================

The :class:`~convis.retina.Retina` model combines filters from
:mod:`convis.filters.retina` into a complete model, similar
to VirtualRetina [Wohrer2009]_.
To load the same xml configuration as VirtualRetina, the 
module :mod:`convis.retina_virtualretina` provides a 
:class:`~convis.retina_virtualretina.RetinaConfiguration` object
and a default configuration.



General Overview
-----------------

The layers that create the model are: 

 - :class:`~convis.filters.retina.OPL`
 - :class:`~convis.filters.retina.Bipolar`
 - :class:`~convis.filters.retina.GanglionInput`
 - :class:`~convis.filters.retina.GanglionSpiking`

The formulas on which the classes are based are:

:class:`~convis.filters.retina.OPL` transforms the input with set of 
linear filters to a center signal :math:`C` and a surround signal :math:`S`
which is subtracted to get the output current :math:`I_{OLP}`: 
$$C(x,y,t) = G * T(wu,Tu) * E(n,t) * L (x,y,t)$$
$$S(x,y,t) = G * E * C(x,y,t)$$ 
$$I_{OLP}(x,y,t) = \\lambda_{OPL}(C(x,y,t) - w_{OPL} S(x,y,t)_)$$ 
:math:`L` is the luminance input,
:math:`G` are spatial (different) filters,
:math:`T` and :math:`E` are temporal filters,
:math:`\lambda_{OPL}` and :math:`w_{OPL}` are scalar weights.
Subscripts of the filters are omitted (:math:`G` instead of :math:`G_C`),
but they should be thought of as each being a specific, unique filter.

Here :math:`*` is a convolution operator.

:class:`~convis.filters.retina.Bipolar` implements contrast gain control with a differential equation:
$$\\frac{dV_{Bip}}{dt} (x,y,t) = I_{OLP}(x,y,t) - g_{A}(x,y,t)dV_{Bip}(x,y,t)$$

 - with :math:`g_{A}(x,y,t) = G * E * Q(V{Bip})(x,y,t)`
 - and :math:`Q(V{Bip}) = g_{A}^{0} + \lambda_{A}V^2_{Bip}`
 - :math:`G` and :math:`E` are again filters

:class:`~convis.filters.retina.GanglionInput` applies a static nonlinearity and another spatial linear filter: 
$$I_{Gang}(x,y,t) = G * N(T * V_{Bip})$$

    - with :math:`N(V) = \frac{i^0_G}{1-\lambda(V-v^0_G)/i^0_G}` if :math:`V < v^0_G`
    - and :math:`N(V) = i^0_G + \lambda(V-v^0_G)` if  :math:`V > v^0_G`

And finally a spiking mechanism :class:`~convis.filters.retina.GanglionSpiking` is implemented by another differential equation: 

$$ \\dfrac{ dV(x,y) }{dt} = I_{Gang}(x,y,t) - g^L V(x,y,t) + \\eta_v(t)$$

    - :math:`V` values above 1 set a refractory variable to a randomly drawn value
    - while the refractory value is larger than 0, :math:`V` of that pixel is set to 0 and the refractory variable is decremented


References
----------

.. [Wohrer2009] Wohrer, A., & Kornprobst, P. (2009).
    Virtual Retina: a biological retina model and simulator, with contrast gain control.
    Journal of Computational Neuroscience, 26(2), 219-49. http://doi.org/10.1007/s10827-008-0108-4


`convis.retina`
------------------------

.. automodule:: convis.retina
   :members:


`convis.filters.retina`
------------------------

.. automodule:: convis.filters.retina
   :members:



`convis.retina_virtualretina`
------------------------------

This module provides compatibility and default configurations
for the convis retina model and the VirtualRetina software 
package.


.. automodule:: convis.retina_virtualretina
   :members:

