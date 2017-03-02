import base
import theano_utils
import theano
import theano.tensor as T
from . import numerical_filters

def simple_difference(a,b,name='error'):
    return base.as_output(a-b,name=name)

def square_difference(a,b,name='error'):
    return base.as_output((a-b)**2,name=name)

def abs_difference(a,b,name='error'):
    return base.as_output(T.abs(a-b),name=name)

def loglikelihood(fr,sp,name='firingrate_loglikelihood'):
    return base.as_output(T.log(fr/(1.0-fr) + 10**-45)*sp + T.log(1.0-fr),name=name)


def van_rossum(sp1,sp2,q=0.5,name='van_rossum_distance'):
    exp_kernel = numerical_filters.m_e_filter(q)
    exp_convolved_sp1 = theano_utils.conv3(base.pad3_txy(sp1,exp_kernel.shape[1]-1,0,0),exp_kernel)
    exp_convolved_sp2 = theano_utils.conv3(base.pad3_txy(sp2,exp_kernel.shape[1]-1,0,0),exp_kernel)
    return base.as_output((exp_convolved_sp1-exp_convolved_sp2)**2,name=name)

def van_rossum_to_firingrate(fr,sp,q=0.5,name='van_rossum_distance'):
    exp_kernel = numerical_filters.m_e_filter(q)
    exp_convolved = theano_utils.conv3(base.pad3_txy(sp,exp_kernel.shape[1]-1,0,0),exp_kernel)
    return base.as_output((fr-exp_convolved)**2,name=name)
