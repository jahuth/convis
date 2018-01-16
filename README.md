# The `convis` package

This python package provides an implementation of the [Virtual Retina](http://www-sop.inria.fr/neuromathcomp/public/software/virtualretina/) developed by Adrien Wohrer. It uses `PyTorch` to simulate spike trains of retinal ganglion cells by directing the input through a number of computation layers. Each layer might do linear or nonlinear computations, eg. convolve the inpute with a spatio-temporal kernel or apply gain control.

TravisCI on the master branch: [![Build Status](https://travis-ci.org/jahuth/convis.svg?branch=master)](https://travis-ci.org/jahuth/convis) 

We are supporting Python 2.7 right now, but are aiming to support Python 3 as well at some point.

Convis is under development and some features might not work in the current master branch or the PyPi releases.
If you discover unexpected behaviour, please leave an Issue on github.

Also there are two mailing lists for people interested in Convis:

 * To recieve announcements of changes, please subsribe to: [convis-users@googlegroups.com](https://groups.google.com/forum/#!forum/convis-users)
 * If you want to participate in the development, please subscribe to: [convis-dev@googlegroups.com](https://groups.google.com/forum/#!forum/convis-dev)



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

To use the model, supply a numpy array as an argument to the `Retina` (for short input) or to the `run` function with a `dt` keyword to split the input in smaller chunks (and automatically reassemble the output):

```python
inp = np.ones((100,20,20))
output = retina(inp)
    
inp = np.ones((2000,20,20))
output = retina.run(inp,dt=100)
```

It will return an `Output` object containing all outputs of the model (the default for retina is two outputs: spikes of On and Off cells).

```python
convis.plot_5d_time(output[0])
title('On Cells (1 line = 1 pixel)')
figure()
convis.plot_5d_matshow(output[0][:,:,::50,:,:])
title('Every 50th frame of activity')
figure()
# dimension 2 is time, so we mean over all others
# to get the average activity
convis.plot_5d_time(output[0].mean((0,1,3,4)))
convis.plot_5d_time(output[1].mean((0,1,3,4)))
title('Mean Activitiy of On and Off Cells')
```

The output object holds the ouput of the computation (in most cases `torch.autograd.Variables`) but can be converted to a `numpy` array with `output.array(...)`. Outputs can be addressed with numbers or names.

```python
>>> type(output[0])
<class 'torch.autograd.variable.Variable'>
>>> type(output[0].data.cpu().numpy())
<class 'numpy.ndarray'>
>>> type(output.array(0))
<class 'numpy.ndarray'>
>>> output[0] is output.ganglion_spikes_ON
True
>>> output[0] is output['ganglion_spikes_ON']
True
>>> np.sum(abs(output.array('ganglion_spikes_ON') - output.array(0)))
0
```

If instead of spikes, only the firing rate should be returned, the retina can be initialized without a spiking mechanism:

```python
retina_without_spikes = convis.retina.Retina(spikes = False)
```

