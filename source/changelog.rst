Changelog
==========

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