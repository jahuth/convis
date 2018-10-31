

.. _doc-models:

Models in `convis.models`
--------------------------

.. automodule:: convis.models
   :members:


Models are essentially the same as filters: they are both
classes that inherit most of their capabilities from :class:`Layer`.
The split into two submodules is an effort to sort them by
complexity / generalizability. Eg. all filters can be combined into
more complex models (combining models does not always make sense)
and all models are ready to "do things", which is not the case for eg.
the :class:`TimePadding` filter.

If you feel the distinction is unnecessary, or you came up with a better
way to separate Layers into submodules, let me know by `submitting a github issue! <https://github.com/jahuth/convis/labels/enhancement>`_

Combinators
^^^^^^^^^^^
These classes can build simple models out of a sequence of Layers
by either combining them sequentially (the output of one layer 
is the input of the next), or in parallel (all layers get the same
input).

.. autoclass:: convis.models.Sequential
   :members:

.. autoclass:: convis.models.List
   :members:

.. autoclass:: convis.models.Parallel
   :members:

Linear-Nonlinear Models 
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: convis.models.L
   :members:

.. autoclass:: convis.models.LN
   :members:

.. autoclass:: convis.models.LNCascade
   :members:

.. autoclass:: convis.models.LNLN
   :members:

.. autoclass:: convis.models.LNFDLNF
   :members:

Convolution Model
^^^^^^^^^^^^^^^^^

.. autoclass:: convis.models.McIntosh
   :members:

Finding all Layers in one submodules `convis.layers` 
-----------------------------------------------------

.. automodule:: convis.layers

:mod:`convis.layers` now contains all :class:`~convis.base.Layers`
from :mod:`convis.filters` and :mod:`convis.models`. If you are 
unsure whether something is only a "filter" or a already a "model", 
they can all be found in the same module now. Still, :mod:`convis.filters` 
and :mod:`convis.models` will continue to be available separately.


.. code-block:: python

    import convis
    len(dir(convis.layers))