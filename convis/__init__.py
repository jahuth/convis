from __future__ import print_function
try:
    import torch
except ImportError:
    print('Please download the torch package from https://pytorch.org')
    raise

__version__ = '0.6.2'
default_grad_enabled = True


def set_steps_per_second(sps):
    """changes the default scaling of temporal constants.

    Does not consistently work retroactively, 
    so if you want to change it, do it before anything else.
    """
    default_resolution.steps_per_second = sps

def set_pixel_per_degree(ppd):
    """changes the default scaling of spatial constants.
    
    Does not consistently work retroactively, 
    so if you want to change it, do it before anything else.
    """
    default_resolution.pixel_per_degree = ppd

def _get_default_grad_enabled():
    return default_grad_enabled

def _get_default_resolution():
    return default_resolution
    
from . import base
from . import variables
default_resolution = variables.ResolutionInfo(10.0,1000.0,1.0,filter_epsilon=0.001)
from . import retina
from . import models
from . import streams
from . import samples
from . import numerical_filters
from . import utils
from . import analysis
from . import kernels
from . import layers

from .base import Layer, prepare_input, Output, Runner
from .variable_describe import help, describe, describe_text, describe_dict, describe_html, animate, animate_to_video
from .variables import Parameter
from .utils import plot, plot_5d_matshow, plot_5d_time


def main():
    import sys
    print(len(sys.argv))

if __name__ == "__main__":
    main()
    sys.exit()
