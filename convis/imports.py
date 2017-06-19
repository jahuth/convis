use_patched_theano = False # for now :/

import numpy
import numpy as np
import matplotlib.pylab as plt
if use_patched_theano:
    from . import patched_theano as theano
    from .patched_theano import tensor as T
else:
    import theano
    import theano.tensor as T

from base import N,  Layer, M, Model, GraphWrapper, describe
from variable_describe import describe