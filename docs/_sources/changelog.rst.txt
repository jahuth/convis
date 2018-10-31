Changelog
==========

0.6.4 (current github version)
------------------------------------

Install with:

.. code-block::
    bash

    pip install git+https://github.com/jahuth/convis.git

Changes: 

  - fixed bug in convolution filter alignment in :class:`~convis.filters.Conv2d` and :class:`~convis.filters.Conv3d`: odd sized filters are now centered correctly
  - fixed two bugs in :class:`~convis.filters.spiking.LeakyIntegrateAndFireNeuron` and :class:`~convis.filters.spiking.RefractoryLeakyIntegrateAndFireNeuron`:
      + The default time step from `convis.default_resolution` was not copied to the internal variable `tau`. Now layers will use the value from the time of their creation.
      + The input was not normalized with the step size, leading to different results when changing the resolution and leak simultaneously.
 - added `flip` argument (default: True) to :meth:`convis.filters.Conv3d.set_weight` to keep the filter and impulse response aligned.
 - fixed bug in plot_impulse (impulse was too long due to padding)
 - added Difference Layer :class:`convis.filters.Diff`
 - added spiking layers :class:`convis.filters.spiking.Poisson` and :class:`convis.filters.spiking.IntegrativeMotionSensor`, which is a DVS like Layer
 - added :class:`convis.streams.ProcessingStream` and :class:`convis.streams.MNISTStream` (and neuromorphic versions: :class:`~convis.streams.PoissonMNISTStream` and :class:`~convis.streams.PseudoNMNIST`)
 - fixed bug in variable :func:`~convis.variable_describe.describe`
 - added usage docs about inputs and outputs


All older versions are available from PyPI. Install the most recent stable version (currently 0.6.3) with:

.. code-block::
    bash

    pip install convis



0.6.3
------

 - :class:`~convis.filters.retina.GanglionSpiking` had a bug in the refractory period
 - the different implementations of OPL Layers now produce roughly the same output
 - FullConvolutionOPLFilter now creates a filter from configuration options
 - :func:`~convis.base.Layer.run()` now always returns an :class:`~convis.base.Output` object
     + fixed a bug where it returned an :class:`~convis.base.Output` object wrapped in a :class:`~convis.base.Output`  object
 - :func:`~convis.base.Layer.run()` can now process infinite streams if it gets an argument `max_t`
 - added warnings to :mod:`convis.streams` classes that are unstable
 - fixed :class:`convis.streams.ImageSequence`
 - :func:`convis.utils.plot_tensor` is now available as `convis.plot_tensor` 


0.6.2
-----

A small amount of bug fixes:

 - Fixed a bug in :class:`convis.filters.retina.GanglionInput`:
    - the convolution filters now initialize without having to reapply a configuration to the layer/parameters
 - made padding flags in :class:`convis.filters.Conv3d` default to True
    - in contrast to PyTorch `Conv3d` the output will now have the same shape as the input *by default*. To disable, use `time_pad=False` and `autopad=False` as arguments.
 - fixed a bug when calling :func:`~convis.base.Output.array` on :class:`convis.base.Output`.
 - fixed a bug where `resolution` was not a valid argument to :func:`convis.filters.Conv3d.gaussian`

0.6.1
-----

 - It is now possible to disable the computational graph see  :ref:`these examples <disable_graph>`
 - The documentation now has :ref:`a section <global_configuration>` about changing the global configuration parameters.
    - `default_resolution` now lives directly in `convis`, no longer in `convis.variables`
    - other submodules now use `convis._get_default_resolution()` to get the current `default_resolution` to avoid copies that don't update
 - some issues with creating inline plots in Python 3 were fixed
 - feeding an :class:`~convis.base.Output` object to a :meth:`~convis.base.Layer.run` function will now take the first output and process it as input
    - this way, it is now easier to continue a graph through multiple models
 - a new submodule :mod:`convis.layers` now contains all :class:`~convis.base.Layers` from :mod:`convis.filters` and :mod:`convis.models`! If you are unsure whether something is only a "filter" or a already a "model", they can all be found in the same module now. Still, :mod:`convis.filters` and :mod:`convis.models` will continue to be available separately.
 - added an unfinished class :class:`convis.models.Dict` which will be similar to :class:`convis.models.List`, but can name its layers arbitrarily.
    - possibly both classes will be merged and replace the :class:`convis.models.Sequential` and :class:`convis.models.Parallel` classes

0.6
---

 - This version fixed an issue with the new PyTorch version (>0.4)