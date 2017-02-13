import base
from base import N, M, GraphWrapper, describe
import retina_base
import filters.retina as retina_filters
import filters.simple as simple_filters
from filters.simple import *

from theano.tensor import as_tensor_variable as as_var
import retina

# just to make sure that we don't use filters.simple version of everything
from base import N, M, GraphWrapper, describe, filter_dbg
