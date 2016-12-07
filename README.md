# The `retina` package

This python package provides an implementation of the [Virtual Retina](http://www-sop.inria.fr/neuromathcomp/public/software/virtualretina/) developed by Adrien Wohrer. It uses `theano` to simulate spike trains of retinal ganglion cells by directing the input through a number of computation nodes. Each node might do linear or nonlinear computations, eg. convolve the inpute with a spatio-temporal kernel or apply gain control.

 * `retina.py` provides a `VirtualRetina` class and corresponding computation nodes
     - the `VirtualRetina` class computes the entire cascade of nodes
     - each node can retain its state, such that input can be chunked
 * `retina_base.py` provides general functions for creating kernels
 * `retina_virtualretina.py` provides default configuration options (as a dictionary) and methods for writing xml configuration files for the original program
 * `vison.py` aims at reimplementing the nodes in a more general framework, such that they can be used for any vision computation.


