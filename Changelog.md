Changelog
==========

See also the more complete changelog [in the documentation](https://jahuth.github.io/convis/changelog.html).

Version 0.6.2
--------------

A small amount of bug fixes:

 - Fixed a bug in `convis.filters.retina.GanglionInput`:
    - the convolution filters now initialize without having to reapply a configuration to the layer/parameters
 - made padding flags in `convis.filters.Conv3d` default to True
    - in contrast to PyTorch `Conv3d` the output will now have the same shape as the input *by default*. To disable, use `time_pad=False` and `autopad=False` as arguments.
 - fixed a bug when calling `.array()` on `convis.base.Output`s.
 - fixed a bug where `resolution` was not a valid argument to `convis.filters.Conv3d.gaussian`

Version 0.6.1
---------------

 - It is now possible to disable the computational graph.
 - some issues with creating inline plots in Python 3 were fixed

Version 0.6
-----------

 - fixed compatibility with PyTorch 0.4.0
    + `convis` is now compatible with PyTorch <= 0.3 and PyTorch 0.4.0
 - VirtualParameters are now also included in the `named_parameter` dictionary and also appear in the `model.p.<tab>` parameter list.
 - the `variables` submodule now offers the functions `zeros`, `zeros_like`, `ones`, `rand` and `randn` which give a autograd.Variable or a Tensor, depending on PyTorch version.