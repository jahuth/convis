from __future__ import print_function

from . import base
from . import retina
from . import models
from . import samples
from . import variables
from . import numerical_filters
from . import utils

from .base import Layer, prepare_input, Output, Runner
from .variable_describe import help, describe, describe_text, describe_dict, describe_html
from .variables import Parameter, default_resolution
from .utils import plot, plot_5d_matshow, plot_5d_time

def main():
    import sys
    print(len(sys.argv))

if __name__ == "__main__":
    main()
    sys.exit()