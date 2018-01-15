import numpy as np
import matplotlib.pylab as plt
from .variables import default_resolution


def conv(a,b,padding_things_equal=[1,3,4],padding_things_tail=[1],*args,**kwargs):
    raise Exception("Needs reimplementation!")
    a_ = a.copy()
    b_ = b.copy()
    a_ = np.pad(a_,[(s-1,s-1) for si,s in enumerate(b_.shape)],mode='constant')
    return _conv_func(a_,b_)


"""
                                                                             
Filters for convolutions
------------------------


"""

def exponential_filter_5d(tau = 0.01, n=0, normalize=True, resolution=None,amplification=1.0,max_length=1000,min_steps=10):
    kernel = exponential_filter_1d(tau=tau,n=n,normalize=normalize,resolution=resolution,amplification=amplification,max_length=max_length,min_steps=min_steps)
    return kernel.reshape((1,len(kernel),1,1,1))

def exponential_filter_1d(tau = 0.01, n=0, normalize=True, resolution=None,amplification=1.0, max_length=1000,min_steps=10,even=None):
    if resolution is None:
        resolution = default_resolution
    tau_in_steps = resolution.seconds_to_steps(tau)
    if n == 0:
        a = amplification/tau_in_steps
        length = min(max(int(-tau_in_steps*np.log(resolution.filter_epsilon/a))+1.0,min_steps),max_length)
        if length <= 1:
            return np.ones(1)
        if even is False and length%2 == 0:
            length += 1
        if even is True and length%2 == 1:
            length += 1
        t = np.linspace(1.0,length,length)
        kernel =  np.exp(-np.linspace(0.4,length-0.6,length)/float(tau_in_steps))
        if normalize:
            kernel *=  a
    else:
        a = amplification
        length = (int(-tau_in_steps*np.log(resolution.filter_epsilon/a))-n)
        if length > max_length:
            length = max_length
        if length <= 1:
            return np.ones(1)
        if even is False and length%2 == 0:
            length += 1
        if even is True and length%2 == 1:
            length += 1
        t = np.linspace(1.0,n*length,n*length)
        kernel = amplification * (n*t)**n * np.exp(-n*t/tau_in_steps) / (np.math.factorial(n-1) * tau_in_steps**(n+1))
    if np.any(np.array(kernel.shape) == 0):
        return np.ones(1)
    return kernel

def exponential_highpass_filter_1d(tau = 0.01, relative_weight=0.1, normalize=True, resolution=None,max_length=1000,min_steps=10):
    if resolution is None:
        return np.ones(1)
    #tau_in_steps = resolution.seconds_to_steps(tau)
    # we amplify the kernel to enforce greater precision
    kernel = -exponential_filter_1d(tau=tau,normalize=normalize,resolution=resolution,amplification=1.0*relative_weight,max_length=max_length,min_steps=min_steps)/1.0
    return np.concatenate([[1], kernel],axis=0)

def exponential_highpass_filter_3d(tau = 0.01, relative_weight=0.1, normalize=True, resolution=None,max_length=1000,min_steps=10):
    kernel = exponential_highpass_filter_1d(tau=tau, relative_weight=relative_weight, normalize=normalize,resolution=resolution,max_length=max_length,min_steps=min_steps)
    return kernel.reshape((len(kernel),1,1))

def exponential_highpass_filter_5d(tau = 0.01, relative_weight=0.1, normalize=True, resolution=None,max_length=1000,min_steps=10):
    kernel = exponential_highpass_filter_1d(tau=tau, relative_weight=relative_weight, normalize=normalize,resolution=resolution,max_length=max_length,min_steps=min_steps)
    return kernel.reshape((1,len(kernel),1,1,1))

def gauss_filter_2d(x_sig,y_sig,normalize=False,resolution=None,minimize=False, even=False, border_factor=1.0):
    """
        A 2d gaussian.

        x_sig and y_sig are the standard deviations in x and y direction.

        if :py:obj:`even` is not None, the kernel will be either made to have even or uneven side lengths, depending on the truth value of :py:obj:`even`.
    """
    if resolution is None:
        resolution = default_resolution
    if x_sig == 0 or y_sig == 0:
        return np.ones((1,1))
    x_sig = resolution.degree_to_pixel(x_sig)
    y_sig = resolution.degree_to_pixel(y_sig)
    a_x = 1.0/(x_sig * np.sqrt(2.0*np.pi))
    x_min = border_factor*np.ceil(np.sqrt(-2.0*(x_sig**2)*np.log(resolution.filter_epsilon/a_x)))
    a_y = 1.0/(y_sig * np.sqrt(2.0*np.pi))
    y_min = border_factor*np.ceil(np.sqrt(-2.0*(y_sig**2)*np.log(resolution.filter_epsilon/a_y)))
    if x_min < 1.0:
        x_gauss = np.ones(2) if even else np.ones(1)
    else:
        X = np.arange(1.0-x_min-(0.5 if even else 0.0),x_min+(0.5 if even else 0.0))
        x_gauss = (a_x *np.exp(-0.5*(X)**2/float(x_sig)**2)).clip(0,1)
    if y_min < 1.0:
        y_gauss = np.ones(2) if even else np.ones(1)
    else:
        Y = np.arange(1.0-y_min-(0.5 if even else 0.0),y_min+(0.5 if even else 0.0))
        y_gauss = (a_y *np.exp(-0.5*(Y)**2/float(y_sig)**2)).clip(0,1)
    kernel = np.prod(np.meshgrid(x_gauss,y_gauss),0)
    if np.any(np.array(kernel.shape) == 0):
        return np.ones((1,1))
    return kernel

def gauss_filter_3d(x_sig,y_sig,normalize=False,resolution=None, even=None,border_factor=1.0):
    """
        A 2d gaussian in a 5d data structure (1 x time x 1 x X x Y)

        x_sig and y_sig are the standard deviations in x and y direction.

        if :py:obj:`even` is not None, the kernel will be either made to have even or uneven side lengths, depending on the truth value of :py:obj:`even`.
    """
    kernel = gauss_filter_2d(x_sig,y_sig,normalize=normalize,resolution=resolution, even=even,border_factor=border_factor)
    return kernel.reshape((1,kernel.shape[0],kernel.shape[1]))

def gauss_filter_5d(x_sig,y_sig,normalize=False,resolution=None, even=None,border_factor=1.0):
    """
        A 2d gaussian in a 5d data structure (1 x time x 1 x X x Y)

        x_sig and y_sig are the standard deviations in x and y direction.

        if :py:obj:`even` is not None, the kernel will be either made to have even or uneven side lengths, depending on the truth value of :py:obj:`even`.
    """
    kernel = gauss_filter_2d(x_sig,y_sig,normalize=normalize,resolution=resolution, even=even,border_factor=border_factor)
    return kernel.reshape((1,1,1,kernel.shape[0],kernel.shape[1]))

def fake_filter(*actual_filters,**kwargs):
    """
        creates a fake filter that contains a single 1. The shape is the shape of all argument filters combined.

        This is needed if the shape of a term has to be adjusted to another term that had originally the same shape, but then got filtered by multiple filters.
    """
    shapes = np.array(np.sum([[aa-1 for aa in a.shape] for a in actual_filters],0) + np.ones(5),dtype=int)
    new_filter = np.zeros(shapes)
    if kwargs.get("centered",True):
        new_filter[:,0,:,int(new_filter.shape[-2]/2),int(new_filter.shape[-1]/2)] = 1
    else:
        new_filter[:,0,:,0,0] = 1
    return new_filter.reshape(shapes)

def fake_filter_shape(*actual_filters):
    """
        returns the shape of all argument filters combined.
    """
    shapes = np.array(np.sum([[aa-1 for aa in a.shape] for a in actual_filters],0) + np.ones(5),dtype=int)
    return shapes

"""
                                                                             
Filters for recursive algorithms
--------------------------------


"""


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
    ek = (1.0-ema)*(1.0-ema) / (1.0+2.0*alpha*ema - ema*ema)
    A1 = ek
    A2 = ek * ema * (alpha-1.0)
    A3 = ek * ema * (alpha+1.0)
    A4 = -ek*ema*ema
    B1 = 2.0*ema
    B2 = -ema*ema
    return {'A1':A1, 'A2':A2, 'A3':A3, 'A4':A4, 'B1':B1, 'B2':B2 }

def sum_kernels(kernels):
    """
        Sums numeric kernels and extends their size 
    """
    max_shape = np.max([k.shape for k in kernels],axis=0)
    new_k = np.zeros(max_shape)
    for k in kernels:
        x1 = np.floor((max_shape[0] - k.shape[0])/2.0)
        x2 = x1 + k.shape[0]
        y1 = np.floor((max_shape[1] - k.shape[1])/2.0)
        y2 = y1 + k.shape[1]
        new_k[x1:x2,y1:y2] += k
    return new_k
