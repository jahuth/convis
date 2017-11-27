.. _pytorch:
PyTorch Basics
===============

PyTorch is a computing library with a focus on deep learning.
It provides three mayor submodules to make deep learning easy:

 * A high performance tensor computing package `torch.tensor`
 * A computational graph that is built while you do your computations `torch.autograd`
 * Classes to package computations into modules and collect parameters hierarchically `torch.nn`


Tensor computing with `torch.tensor`
----------------------------------------



Automated differentiation with `torch.autograd`
-------------------------------------------------




Model building with `torch.nn`
------------------------------


PyTorch Extensions in Convis
===============================



Layer
---------

Layers are extensions of `torch.nn.Module`s. They behave very similarly, but have a few additional features:

 * a Layer knows if it accepts 1d, 3d, or 5d time sequence input and can broadcast the input accordingly if it has too few dimensions
 * instead of running the model on the complete time series, the input can be automatically chunked by using the `.run(.., dt=chunk_length)` method instead of calling the Layer directly.
 * a Layer can create its own optimizer



Output
---------

A class that collects all outputs of a Layer for Layers that have more than one output.



Conv3d
---------

The extended Conv3d operation can automatically pad input in x and y dimension.