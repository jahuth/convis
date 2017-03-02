import misc_utils

import theano_utils
reload(theano_utils)
import variable_describe
reload(variable_describe)

import base
reload(base)
from base import N, M, GraphWrapper, describe
import retina_base
reload(retina_base)
import filters.retina as retina_filters
reload(filters.retina)
import filters.simple as simple_filters
reload(filters.simple)
from filters.simple import *

from theano.tensor import as_tensor_variable as as_var
import retina
reload(retina)
import models

import error_functions
reload(error_functions)

# just to make sure that we don't use filters.simple version of everything
from base import N, M, GraphWrapper, describe, filter_dbg
