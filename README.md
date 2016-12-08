# The `convis` package

This python package provides an implementation of the [Virtual Retina](http://www-sop.inria.fr/neuromathcomp/public/software/virtualretina/) developed by Adrien Wohrer. It uses `theano` to simulate spike trains of retinal ganglion cells by directing the input through a number of computation nodes. Each node might do linear or nonlinear computations, eg. convolve the inpute with a spatio-temporal kernel or apply gain control.

Usage Example:

```python
import convis

c = convis.retina.RetinaConfiguration()
ret = convis.retina.Retina(c)
ret.create_function()
inp = np.zeros((200,50,50))
inp[:,20:30,20:30] = 255.0*(rand(*inp[:,20:30,20:30].shape)<0.2)
out = ret.run(inp)

plot(np.mean(out[0],(1,2)))
plot(np.mean(out[1],(1,2)))
```

An older version was published as <a href="https://github.com/jahuth/retina">the retina package</a>


