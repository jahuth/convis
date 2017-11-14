from __future__ import print_function
#from . import misc_utils

#from . import variables
#from . import imports
#imports.T._variable_factory = variables.make_variable

#from . import theano_utils
#from . import variable_describe

#from .base import N, M, GraphWrapper, describe
#from . import retina_base
#from .filters import retina as retina_filters
#from .filters import simple as simple_filters
#from .filters.simple import *

from . import base
from . import retina
from . import models
from . import samples
from . import variables
from . import numerical_filters

#from . import error_functions

# just to make sure that we don't use filters.simple version of everything
from .base import Layer
from .variable_describe import help, describe, describe_text, describe_dict, describe_html
from .variables import Parameter, default_resolution



def main():
    import sys
    print(len(sys.argv))

if __name__ == "__main__":
    main()
    sys.exit()