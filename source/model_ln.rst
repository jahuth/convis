.. _model_ln:

Linear-Nonlinear Models
------------------------

Usage
~~~~~~~~~~~~~~~~~~~

The linear-nonlinear models available include:

 * `L` a simple linear model
 * `LN` a simple linear-nonlinear model
 * `LNLN`/`LNSN` a linear-nonlinear cascade (or subunit) model
 * `LNFDSNF` linear-nonlinear subunit model with feedback and individual delays

There is also :class:`convis.models.LNCascade`, a model implementation that makes it easier to add and modify layers::

    >>> m = convis.models.LNCascade(n=2, nonlinearity=convis.filters.NLRectify()) # create two convolution layers
    >>> m.add_layer(linear = convis.filters.RF, nonlinear=lambda x: x)

By default, convolution models will give a population activity of cells distributed over the whole image.
If you want to fit the receptive field of a single cell, you might want to use a :class:`convis.filters.RF` linear filter
instead of the :class:`convis.filters.Conv3d` filter. A :class:`~convis.filters.RF` filter will output
a single time series (or multiple if output channels are > 1). The spatial extent of the filter should
match your input image to get a meaningful receptive field.