Changelog
==========

See also the more complete changelog [in the documentation](https://jahuth.github.io/convis/changelog.html).

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