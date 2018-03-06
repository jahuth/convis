from __future__ import print_function
try:
    import torch
except ImportError:
    print('Please download the torch package from https://pytorch.org')
    raise

__version__ = '0.5.2.0'

from . import base
from . import retina
from . import models
from . import samples
from . import streams
from . import variables
from . import numerical_filters
from . import utils
from . import analysis
from . import kernels

from .base import Layer, prepare_input, Output, Runner
from .variable_describe import help, describe, describe_text, describe_dict, describe_html, animate, animate_to_video
from .variables import Parameter, default_resolution
from .utils import plot, plot_5d_matshow, plot_5d_time

def main():
    import sys
    print(len(sys.argv))

if __name__ == "__main__":
    main()
    sys.exit()