import misc_utils

import variables
import imports
#imports.T._variable_factory = variables.make_variable

import theano_utils
import variable_describe

import base
from base import N, M, GraphWrapper, describe
import retina_base
import filters.retina as retina_filters
import filters.simple as simple_filters
from filters.simple import *

import retina
import models

import error_functions

# just to make sure that we don't use filters.simple version of everything
from base import N, M, GraphWrapper, describe, filter_dbg
from variable_describe import help, describe, describe_text, describe_dict, describe_html

import samples
import kernels

from imports import theano, T

def main():
    import sys
    print len(sys.argv)

if __name__ == "__main__":
    main()
    sys.exit()