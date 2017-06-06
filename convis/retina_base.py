"""

This module contains base classes and useful functions for the retina module.

"""


import litus
from .imports import theano
from .imports import T
import numpy as np
import matplotlib.pylab as plt
from .theano_utils import conv3d, conv2d
import uuid 

dtensor5 = T.TensorType('float64', (False,)*5)

A = dtensor5('A')
B = dtensor5('B')
C = conv3d(A,B)
_conv_func = theano.function(inputs=[A,B], outputs=C)

def conv(a,b,padding_things_equal=[1,3,4],padding_things_tail=[1],*args,**kwargs):
    a_ = a.copy()
    b_ = b.copy()
    a_ = np.pad(a_,[(s-1,s-1) for si,s in enumerate(b_.shape)],mode='constant')
    return _conv_func(a_,b_)


### Functions to create filters 
#

def minimize_filter(f,filter_epsilon = 0.0, minimize_xy=True, minimize_t_tail=True, minimize_t_start=False):
    """
        reduces a filter by taking of the sides if they are smaller than :py:obj:`filter_epsilon`
    """
    if np.max(np.abs(f)) <= filter_epsilon:
        return np.array([0]).reshape((1,1,1,1,1))
    if minimize_xy:
        if f.shape[3] > 1:
            while np.sum(np.abs(f[:,:,:,0,:])) + np.sum(np.abs(f[:,:,:,-1,:])) <= filter_epsilon*2.0:
              f = f[:,:,:,1:-1,:]
        if f.shape[4] > 1:
            while np.sum(np.abs(f[:,:,:,:,0])) + np.sum(np.abs(f[:,:,:,:,-1])) <= filter_epsilon*2.0:
                f = f[:,:,:,:,1:-1]
    if f.shape[1] > 1:
        if minimize_t_start:
            while np.sum(np.abs(f[:,0,:,:])) <= filter_epsilon:
                f = f[:,1:,:,:,:]
        if minimize_t_tail:
            while np.sum(np.abs(f[:,-1,:,:])) <= filter_epsilon:
                f = f[:,:-1,:,:,:]
    return f.copy()

def m_t_filter(steepness=2.0,relative_weight=1.0,normalize=True,minimize=True,retina=None,epsilon=0.0001):
    """
        A T filter creates a transient response by combining a sharp positive (1) term with a negative exponential that has area :py:obj:`relative_weight`.
        The sum of the area of this filter is 1-:py:obj:`relative_weight`.
    """
    if retina is not None:
        steepness = retina.seconds_to_steps(steepness)
    length = 5.0*steepness
    T_filter = np.exp(-np.arange(0.0,length)/float(steepness))
    if normalize:
        T_filter = T_filter/np.sum(T_filter)
    if minimize:
        T_filter= minimize_filter(T_filter.reshape((1,len(T_filter),1,1,1)),epsilon)[0,:,0,0,0]
    if normalize:
        T_filter = T_filter/np.sum(T_filter)
    if len(T_filter) == 1:
        T_filter = np.array([1.0])
    T_filter = -1.0*relative_weight * T_filter
    T_filter = np.array([1] + T_filter.tolist())
    return T_filter.reshape((1,len(T_filter),1,1,1))


def m_e_filter(steepness=2.0,normalize=True,minimize=True,retina=None,epsilon=0.001):
    """
        An exponential filter.
    """
    if retina is not None:
        steepness = retina.seconds_to_steps(steepness)
    length = 10.0*steepness
    kernel = np.exp(-np.arange(0.0,length)/float(steepness))
    if normalize:
        kernel = kernel/np.sum(kernel)
    kernel = kernel.reshape((1,len(kernel),1,1,1))
    if minimize:
        kernel = minimize_filter(kernel,epsilon)
    return kernel
    
def m_en_filter(n,tau=2.0,normalize=True,minimize=True,retina=None,epsilon=0.001):
    """
        Creates an exponential cascade filter that is convolved :py:obj:`n` times.

        :py:obj:`normalize` does nothing as the kernels are already generated normalized.
    """
    if n == 0:
        return m_e_filter(tau,normalize=normalize,minimize=minimize,retina=retina,epsilon=epsilon)
    if retina is not None:
        tau = retina.seconds_to_steps(tau)
    length = int(4.0*tau)
    t = np.linspace(0.0,length,length)
    kernel = (n*t)**n * np.exp(-n*t/tau) / (np.math.factorial(n-1) * tau**(n+1))
    if normalize:
        return kernel.reshape((1,len(kernel),1,1,1))/np.sum(kernel)
    if minimize:
        return minimize_filter(kernel.reshape((1,len(kernel),1,1,1)),epsilon)
    return kernel.reshape((1,len(kernel),1,1,1))

def m_g_filter(x_sig,y_sig,normalize=False,retina=None,minimize=False,epsilon=0.00001, even=None, make_uneven=False):
    """
        A 2d gaussian in a 5d data structure (1 x time x 1 x X x Y)

        x_sig and y_sig are the standard deviations in x and y direction.

        if :py:obj:`even` is not None, the kernel will be either made to have even or uneven side lengths, depending on the truth value of :py:obj:`even`.
    """
    if retina is not None:
        x_sig = retina.degree_to_pixel(x_sig)
        y_sig = retina.degree_to_pixel(y_sig)
    x = np.ceil(x_sig*5.0)+1.0
    y = np.ceil(y_sig*5.0)+1.0
    if x_sig == 0 or y_sig == 0:
        return np.array([1.0]).reshape((1,1,1,1,1))
    if even is not None:
        if even:
            if x%2 == 1:
                x += 1
            if y%2 == 1:
                y += 1    
        if not even:
            if x%2 == 0:
                x += 1
            if y%2 == 0:
                y += 1 
    x_gauss = np.exp(-(1.0/float(x_sig)**2)*(np.arange(-np.floor(x/2.0)+(0.5 if even else 0.0),np.ceil(x/2.0)+(0.5 if even else 0.0)))**2)
    y_gauss = np.exp(-(1.0/float(y_sig)**2)*(np.arange(-np.floor(y/2.0)+(0.5 if even else 0.0),np.ceil(y/2.0)+(0.5 if even else 0.0)))**2)
    kernel = np.prod(np.meshgrid(x_gauss,y_gauss),0).reshape((1,1,1,len(y_gauss),len(x_gauss)))
    if normalize:
        kernel = kernel/np.sum(kernel)
    if minimize:
        kernel = minimize_filter(kernel,epsilon)
    return kernel


def m_g_filter_2d(x_sig,y_sig,normalize=False,retina=None,minimize=False,epsilon=0.00001, even=None, make_uneven=False):
    """
        A 2d gaussian.

        x_sig and y_sig are the standard deviations in x and y direction.

        if :py:obj:`even` is not None, the kernel will be either made to have even or uneven side lengths, depending on the truth value of :py:obj:`even`.
    """
    if retina is not None:
        x_sig = retina.degree_to_pixel(x_sig)
        y_sig = retina.degree_to_pixel(y_sig)
    x = x_sig*5.0
    y = y_sig*5.0
    if x_sig == 0 or y_sig == 0:
        return np.array([1]).reshape((1,1))
    if even is not None:
        if even:
            if x%2 == 1:
                x += 1
            if y%2 == 1:
                y += 1    
        if not even:
            if x%2 == 0:
                x += 1
            if y%2 == 0:
                y += 1 
    x_gauss = np.exp(-(1.0/float(x_sig)**2)*(np.arange(-np.floor(x/2.0)+(0.5 if even else 0.0),np.ceil(x/2.0)+(0.5 if even else 0.0)))**2)
    y_gauss = np.exp(-(1.0/float(y_sig)**2)*(np.arange(-np.floor(y/2.0)+(0.5 if even else 0.0),np.ceil(y/2.0)+(0.5 if even else 0.0)))**2)
    kernel = np.prod(np.meshgrid(x_gauss,y_gauss),0)
    if normalize:
        kernel = kernel/np.sum(kernel)
    if minimize:
        kernel = minimize_filter(kernel,epsilon)
    return kernel

def fake_filter(*actual_filters,**kwargs):
    """
        creates a fake filter that contains a single 1. The shape is the shape of all argument filters combined.

        This is needed if the shape of a term has to be adjusted to another term that had originally the same shape, but then got filtered by multiple filters.
    """
    shapes = np.array(np.sum([[aa-1 for aa in a.shape] for a in actual_filters],0) + np.ones(5),dtype=int)
    new_filter = np.zeros(shapes)
    if kwargs.get("centered",True):
        #print len(new_filter)/2
        new_filter[:,0,:,new_filter.shape[-2]/2,new_filter.shape[-1]/2] = 1
        #new_filter[-1] = 1
    else:
        #new_filter[0] = 1
        new_filter[:,0,:,0,0] = 1
    return new_filter.reshape(shapes)
def _depr_fake_filter(*actual_filters):
    shapes = np.prod([a.shape for a in actual_filters],0)
    new_filter = np.zeros(shapes).flatten()
    new_filter[0] = 1
    return new_filter.reshape(shapes)

def fake_filter_shape(*actual_filters):
    """
        returns the shape of all argument filters combined.
    """
    shapes = np.array(np.sum([[aa-1 for aa in a.shape] for a in actual_filters],0) + np.ones(5),dtype=int)
    return shapes

def find_nonconflict(name,names):
    if name in names:
        i = 0
        while name+' '+str(i) in names:
            i+=1
        names.append(name+' '+str(i))
        return name
    else:
        names.append(name)
        return name

def ab_filter_exp(tau,step = 0.001):
    """ create an Exp filter and return arrays for the coefficients 

    TODO: describe how to use a and b

    """
    if tau < 0:
        raise Exception("Negative time constants are not implemented")
    if tau == 0:
        a = np.array([1.0])
        b = np.array([])
        return [a,b]
    a = np.array([1.0,-np.exp(-step/tau)])
    b = np.array([1.0-np.exp(-step/tau)])
    return [a,b]

def ab_filter_exp_cascade(self,tau, n, step = 0.001):
    """ create an ExpCascade filter and return arrays for the coefficients """
    from scipy.misc import comb
    tau = float(tau)
    n = int(n)
    if tau < 0:
        raise Exception("Negative time constants are not implemented")
    if n < 0:
        raise Exception("Negative cascade number is not allowed")
    if tau == 0:
        a = np.array([1.0])
        b = np.array([])
        return [a,b]
    tauC = tau/float(n) if n > 0 else tau
    c = np.exp(-step/tauC)
    N = n + 1
    a = np.array([(-c)**(i)*comb(N,i) for i in range(N+1)])
    b = np.array([(1.0-c)**N])
    return [a,b]

def concatenate_time(a,b):
    """
        concatenates ndarrays on the 'typical' time axis used in this implementation:

            If the objects are 1d, they are concatenated along this axis.
            (2d are static images only)
            If the data is 3d, then the first axis is time.
            If the data is 4d, then it is assumed that it is a collection of 3d objects, thus time is at the second index.
            5d Objects are usually used as filters, while the first and third index are unused. Time is the second dimension.
            6d objects are assumed to be collections of 5d objects such that time is at the third index.

        If the data is a list or a tuple, elements from both a and b are zipped and concatenated along time.

    """
    if type(a) in [list,tuple] and type(b) in [list,tuple]:
        return [concatenate_time(a_,b_) for (a_,b_) in zip(a,b)]
    if len(a.shape) == 1 and len(b.shape) == 1:
        return np.concatenate([a,b],0)
    if len(a.shape) == 3 and len(b.shape) == 3:
        return np.concatenate([a,b],0)
    if len(a.shape) == 4 and len(b.shape) == 4:
        return np.concatenate([a,b],1)
    if len(a.shape) == 5 and len(b.shape) == 5:
        return np.concatenate([a,b],1)
    if len(a.shape) == 6 and len(b.shape) == 6:
        return np.concatenate([a,b],2)

def deriche_coefficients(density):
    """
       Creates deriche coefficients for a given map of filter density 
    """
    alpha = 1.695 * density
    ema = np.exp(-alpha)
    ek = (1-ema)*(1-ema) / (1+2*alpha*ema - ema*ema)
    A1 = ek
    A2 = ek * ema * (alpha-1.0)
    A3 = ek * ema * (alpha+1.0)
    A4 = -ek*ema*ema
    B1 = 2*ema
    B2 = -ema*ema
    return {'A1':A1, 'A2':A2, 'A3':A3, 'A4':A4, 'B1':B1, 'B2':B2 }