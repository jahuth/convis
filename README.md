# The `convis` package

This python package provides an implementation of the [Virtual Retina](http://www-sop.inria.fr/neuromathcomp/public/software/virtualretina/) developed by Adrien Wohrer. It uses `PyTorch` to simulate spike trains of retinal ganglion cells by directing the input through a number of computation layers. Each layer might do linear or nonlinear computations, eg. convolve the inpute with a spatio-temporal kernel or apply gain control.

TravisCI on the master branch: [![Build Status](https://travis-ci.org/jahuth/convis.svg?branch=master)](https://travis-ci.org/jahuth/convis) 

We are supporting Python 2.7 right now, but are aiming to support Python 3 as well at some point.

Convis is under development and some features might not work in the current master branch or the PyPi releases.
If you discover unexpected behaviour, please leave an Issue on github.

Also there are two mailing lists for people interested in Convis:

 * To recieve announcements of changes, please subsribe to: convis-users@googlegroups.com
 * If you want to participate in the development, please subscribe to: convis-dev@googlegroups.com 



Usage Example:

```python
import convis

c = convis.retina.RetinaConfiguration()
ret = convis.retina.Retina(c)
inp = np.zeros((2000,50,50))
inp[:,20:30,20:30] = 255.0*(rand(*inp[:,20:30,20:30].shape)<0.2)
out = ret.run(inp)

plot(np.mean(out[0],(0,1,3,4)))
plot(np.mean(out[1],(0,1,3,4)))
```

An earlier version using `theano` has been put on hold, but is still available [here](http://github.com/jahuth/convis_theano). If you are interested in continued development of the `theano` version, please let me know!
An even older version was published as <a href="https://github.com/jahuth/retina">the retina package</a>

## Installation

Installing `convis` and `PyTorch` itself is not complicated.

Requirements for the base installation are: Python 2.7 or Python 3.5, Numpy, SciPy.

```bash
pip install convis
```

or install the latest version from github:

```bash
pip install git+https://github.com/jahuth/convis.git
```

or clone the repository and install it locally:

```bash
git clone https://github.com/jahuth/convis.git
# change something in the source code
pip install -e convis
```


I recommend installing opencv, and jupyter notebook, if you do not already have it installed:

```bash
pip install convis notebook
# eg. for ubuntu:
sudo apt-get install python-opencv
```

## Introduction

`convis` provides a retina model which is identical to `VirtualRetina` and tools
to create either simpler or more complex models.

A preprint of a paper about `convis` is available at bioarxiv: 
["Convis: A Toolbox To Fit and Simulate Filter-based Models of Early Visual Processing"](https://doi.org/10.1101/169284).

## The Retina model

A description of all parameters for the retina model can be obtained directly from
an instantiated model. This description contains doc strings for each parameter.
```python
import convis
retina = convis.retina.Retina()
print(retina)
```

Here is a graph of the model:
<a href="retina_graph.png"><img src="retina_graph.png" widht="200"/></a>

To use the model, supply a numpy array as an argument to the `Retina` (for short input) or to the `run_in_chunks` function:

```python
inp = np.ones((100,20,20))
output = retina(inp)

inp = np.ones((10000,20,20))
output = retina.run_in_chunks(inp,200)
```

It will return an object containing all outputs of the model (the default for retina is two outputs: spikes of On and Off cells).

If instead of spikes, only the firing rate should be returned, the retina can be initialized without a spiking mechanism:

```python
retina_without_spikes = convis.retina.Retina(spikes = False)
```

