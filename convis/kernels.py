
from base import GraphWrapper
from .imports import theano
from .imports import T
import numpy as np
from misc_splines import create_splines_linspace, create_splines_logspace
from variables import as_parameter, as_variable, default_resolution
from . import theano_utils

raise_on_mismatch = False

def DenseKernel1d(t=20,name='dense_kernel'):
    if isinstance(t,GraphWrapper):
        t = t.compute()
    if type(t) is int:
        t = np.zeros(t)
    if not len(t.shape) == 1:
        raise Exception('1d Kernel can only have one dimension!')
    return DenseKernel(t,name=name)

def DenseKernel2d(x=10,y=10,name='dense_kernel'):
    if isinstance(t,GraphWrapper):
        t = t.compute()
    if type(x) is int:
        x = np.zeros((x,y))
    if not len(x.shape) == 2:
        raise Exception('2d Kernel can only have two dimensions!')
    return DenseKernel(x,name=name)

def DenseKernel3d(t=20,x=10,y=10,name='dense_kernel'):
    if isinstance(t,GraphWrapper):
        t = t.compute()
    if type(t) is int:
        t = np.zeros((t,x,y))
    if not len(t.shape) == 3:
        raise Exception('3d Kernel can only have three dimensions!')
    return DenseKernel(t,name=name)

def DenseKernel5d(t=20,x=10,y=10,name='dense_kernel'):
    if isinstance(t,GraphWrapper):
        t = t.compute()
    if type(t) is int:
        t = np.zeros((t,x,y))
    if not len(t.shape) == 5:
        raise Exception('5d Kernel can only have five  dimensions!')
    return DenseKernel(t,name=name)

def DenseKernel(kernel,name='dense_kernel'):
    """
        Creates a DenseKernel parameter from a numpy array.

        Can be used to replace parametrized kernels.
    """
    return convis.as_parameter(theano.shared(kernel),name=name)

class SplineKernel1d(GraphWrapper):
    def __init__(self,length=20,n=5,name='spline_kernel',log=True,approximate=None,remove_last_spline=True):
        """
            A 1 dimensional kernel made from `n` splines of length `t` bins.

            Splines can be logarithmic (default) or linear.

            This object can be used to replace one dimensional kernel parameters::

                temporal_filter = convis.filters.simple.K_1d_kernel_filter()
                temporal_filter.parameters.kernel = SplineKernel1d(20,5)

            Instead of one parameter `kernel`, the coefficients and splines are now the parameters::

                convis.describe(temporal_filter.parameters.kernel.coefficients)

            After the kernel was created, it can still approximate arbitrary other filters::

                temporal_filter.parameters.kernel._.approximate(rand(20))

            The splines have dimensions time x spline.
        """
        if n <= 2:
            raise Exception('n has to be 3 or greater!')
        if log:
            splines = as_variable(theano.shared(create_splines_logspace(length,n-2,remove_last_spline)),name='splines')
        else:
            splines = as_variable(theano.shared(create_splines_linspace(length,n-2,remove_last_spline)),name='splines')
        coefficients = as_parameter(theano.shared(np.ones(n)),name='coefficients')
        kernel = T.sum(splines.dimshuffle(0,1)*coefficients.dimshuffle('x',0),axis=1)
        super(SplineKernel1d,self).__init__(kernel,name=name)
        if approximate is not None:
            self.approximate(approximate)
    def approximate(self,target):
        """
            Approximates a `target` by setting the coefficients such that their sum
            approaches `target`.

            When convis.kernel.raise_on_mismatch is False (current default),
            a longer target will be truncated and a short one will be padded with zeros. 
        """
        if isinstance(target, GraphWrapper):
            target = target.compute()
        if hasattr(target,'get_value'):
            target = target.get_value()
        splines = self.variables.splines.get_value()
        global raise_on_mismatch
        if raise_on_mismatch and target.shape[0] != splines.shape[0]:
            raise Exception('Dimensions of target does not match splines! (Deactivate this exception by setting convis.kernel.raise_on_mismatch to False)')
        if target.shape[0] < splines.shape[0]:
            new_target = np.zeros(splines.shape[0])
            new_target[:target.shape[0]] = target
            target = new_target
        cfs = np.dot(target[:splines.shape[0]],splines)
        self.variables.coefficients.set_value(cfs/np.sum(splines,axis=0).clip(0.0000001,None))
    def compute(self):
        f = theano.function([],self.graph)
        return f()

class SplineKernel2d(GraphWrapper):
    def __init__(self,length=20,n_x=5,n_y=5,name='spline_kernel',approximate=None):
        """
            A 2 dimensional kernel made from `n_x` times `n_y` splines.
    
        """
        if n_x <= 2 or n_y <= 2:
            raise Exception('n has to be 3 or greater!')
        splines_x = create_splines_linspace(20,n_x-2,False)[:,1:-1].transpose()
        splines_y = create_splines_linspace(20,n_y-2,False)[:,1:-1].transpose()
        splines_xy = []
        for x in splines_x:
            for y in splines_y:
                splines_xy.append(x[:,np.newaxis] * y[np.newaxis,:])
        splines_xy = np.array(splines_xy)
        spatial_splines = as_variable(theano.shared(splines_xy),name='splines') # spline_number,x,y
        coefficients = as_parameter(theano.shared(np.ones(splines_xy.shape[0])),name='coefficients')
        kernel = T.sum(spatial_splines*coefficients.dimshuffle(0,'x','x'),axis=0)
        super(SplineKernel2d,self).__init__(kernel,name=name)
        if approximate is not None:
            self.approximate(approximate)
    def approximate(self,target):
        """
            Approximates a `target` by setting the coefficients such that their sum
            approaches `target`.

            When convis.kernel.raise_on_mismatch is False (current default),
            a longer target will be truncated and a short one will be padded with zeros. 
        """
        if isinstance(target, GraphWrapper):
            target = target.compute()
        if hasattr(target,'get_value'):
            target = target.get_value()
        splines = self.variables.splines.get_value()
        global raise_on_mismatch
        if raise_on_mismatch and (target.shape[0] != splines.shape[1] or target.shape[1] != splines.shape[2]):
            raise Exception('Dimensions of target does not match splines! (Deactivate this exception by setting convis.kernel.raise_on_mismatch to False)')
        target = target[:splines.shape[1],:splines.shape[2]]
        if target.shape[0] < splines.shape[1] or target.shape[1] < splines.shape[2]:
            new_target = np.zeros((splines.shape[1],splines.shape[2]))
            new_target[:target.shape[0],:target.shape[1]] = target[:splines.shape[1],:splines.shape[2]]
            target = new_target
        flat_splines = splines.reshape(splines.shape[0],-1).transpose()
        cfs = np.dot(target.reshape(target.shape[0]*target.shape[1]),
             flat_splines)
        self.variables.coefficients.set_value(cfs/np.sum(flat_splines,axis=0).clip(0.0000001,None))
    def compute(self):
        f = theano.function([],self.graph)
        return f()

class SplineKernel3d(GraphWrapper):
    """
        Crazy 3d splines! (but maybe less crazy than a 3d dense kernel? who knows...)
    """
    pass

class ExponentialKernel1d(GraphWrapper):
    def __init__(self,length=20,tau=2,n=0,name='exponetial_kernel',approximate=None,resolution=None):
        """

            Creates a (cascade) exponential kernel.

                length:
                    minimal length of the kernel
                tau:
                    time constant of the kernel (time until the signal shrinks to 1/e)
                n:
                    number of cascades. Default: n=0 (simple exponential kernal)

            The minimal length of the filter is 4 times the time constant.
            If length is shorter than that, it will be ignored.

        """
        if resolution is None:
            resolution = default_resolution
        tau = as_parameter(tau,name='tau')*resolution.var_steps_per_second
        length = T.max([as_parameter(length,name='length')*resolution.var_steps_per_second,4*tau])
        t_range = T.arange(length)
        n = as_parameter(n,name='n')
        factorial = theano.tensor.gamma
        kernel = theano.ifelse.ifelse(T.eq(n,0),
                                T.exp(-t_range/tau)/tau,
                               (n*t_range)**n * T.exp(-n*t_range/tau) / (factorial(n) * tau**(n+1)))
                               #(n*t_range)**n * T.exp(-n*t_range/tau) / (factorial(n-1) * tau**(n+1)))
        super(ExponentialKernel1d,self).__init__(kernel,name=name)
        if approximate is not None:
            self.approximate(approximate)
    def approximate(self,target):
        # TODO
        raise Exception('Not yet implemented!')
    def compute(self):
        f = theano.function([],self.graph)
        return f()

class ExponentialHighPassKernel1d(GraphWrapper):
    def __init__(self,length=20,tau=2,n=0,relative_weight=1.0,name='exponetial_kernel',approximate=None,resolution=None):
        """

            Creates a negative (cascade) exponential kernel, precedet by a positive 1.

                length:
                    minimal length of the kernel
                tau:
                    time constant of the kernel (time until the signal shrinks to 1/e)
                n:
                    number of cascades. Default: n=0 (simple exponential kernal)

            The minimal length of the filter is 4 times the time constant.
            If length is shorter than that, it will be ignored.

        """
        if resolution is None:
            resolution = default_resolution
        tau = as_parameter(tau,name='tau')*resolution.var_steps_per_second
        relative_weight = as_parameter(relative_weight,name='relative_weight')
        length = T.max([as_parameter(length,name='length')*resolution.var_steps_per_second,4*tau])
        t_range = T.arange(length)
        n = as_parameter(n,name='n')
        factorial = theano.tensor.gamma
        kernel = T.concatenate([[1.0],-relative_weight*theano.ifelse.ifelse(T.eq(n,0),
                                T.exp(-t_range/tau)/tau,
                               (n*t_range)**n * T.exp(-n*t_range/tau) / (factorial(n) * tau**(n+1)))],axis=0)
        super(ExponentialHighPassKernel1d,self).__init__(kernel,name=name)
        if approximate is not None:
            self.approximate(approximate)
    def approximate(self,target):
        # TODO
        raise Exception('Not yet implemented!')
    def compute(self):
        f = theano.function([],self.graph)
        return f()

class GaussKernel2d(GraphWrapper):
    def __init__(self,x_sig,y_sig,epsilon=0.001,name='spline_kernel', even=False,resolution=None):
        """
            Very simple gauss filter (aligned with x and y axis).

            For a fancy version see `TiltedGaussKernel2d`.
        """
        if resolution is None:
            resolution = default_resolution
        x_sig = as_parameter(x_sig,name='x_sig')*resolution.var_pixel_per_degree
        y_sig = as_parameter(y_sig,name='y_sig')*resolution.var_pixel_per_degree
        epsilon = as_parameter(epsilon,name='epsilon')
        a_x = 1.0/(x_sig * T.sqrt(2.0*np.pi))
        x_min = T.ceil(T.sqrt(-2.0*(x_sig**2)*T.log(epsilon/a_x)))
        a_y = 1.0/(y_sig * T.sqrt(2.0*np.pi))
        y_min = T.ceil(T.sqrt(-2.0*(y_sig**2)*T.log(epsilon/a_y)))
        X = T.arange(1.0-x_min-(0.5 if even else 0.0),x_min+(0.5 if even else 0.0))
        x_gauss = (a_x * T.exp(-0.5*(X)**2/x_sig**2)).clip(0,1)
        Y = T.arange(1.0-y_min-(0.5 if even else 0.0),y_min+(0.5 if even else 0.0))
        y_gauss = (a_y *T.exp(-0.5*(Y)**2/y_sig**2)).clip(0,1)
        kernel = T.outer(x_gauss,y_gauss)
        super(GaussKernel2d,self).__init__(kernel,name=name)
    def compute(self):
        f = theano.function([],self.graph)
        return f()


class TiltedGaussKernel2d(GraphWrapper):
    def __init__(self,x_sig,y_sig,x_offset=0,y_offset=0,phi=0,epsilon=0.001,name='spline_kernel', even=False,resolution=None):
        """
            Allows a 2d gaussian to be tilted and offset from the center.

            `x_sig` and `y_sig` are the width of the gaussian in the two principal directions.

            `phi` is the rotation in rad. when `phi` = 0, x_sig aligns with the x axis,
            y_sig aligns with the y axis.

            The size of the filter is determined automatically and is always square.
            The offsets are from the center, ie. with each increase in x_offset, the filter
            grows by twice that amount.

        """
        if resolution is None:
            resolution = default_resolution
        x_sig = as_parameter(x_sig,name='x_sig')*resolution.var_pixel_per_degree
        y_sig = as_parameter(y_sig,name='y_sig')*resolution.var_pixel_per_degree
        x_offset = as_parameter(x_offset,name='x_offset')
        y_offset = as_parameter(y_offset,name='y_offset')
        phi = as_parameter(phi,name='phi')
        epsilon = as_parameter(epsilon,name='epsilon')
        a_x = 1.0/(x_sig * T.sqrt(2.0*np.pi))
        x_min = T.ceil(T.sqrt(-2.0*(x_sig**2)*T.log(epsilon/a_x)))
        a_y = 1.0/(y_sig * T.sqrt(2.0*np.pi))
        y_min = T.ceil(T.sqrt(-2.0*(y_sig**2)*T.log(epsilon/a_y)))
        length = T.max([x_min,y_min]) + T.max([abs(x_offset),abs(y_offset)])
        X = T.arange(1.0-length-(0.5 if even else 0.0),length+(0.5 if even else 0.0))
        Y = T.arange(1.0-length-(0.5 if even else 0.0),length+(0.5 if even else 0.0))
        u = T.sin(phi)*(X.dimshuffle(0,'x')+x_offset) + T.cos(phi)*(Y.dimshuffle('x',0)+y_offset)
        v = T.cos(phi)*(X.dimshuffle(0,'x')+x_offset) - T.sin(phi)*(Y.dimshuffle('x',0)+y_offset)
        kernel = (a_x * T.exp(-0.5*(u)**2/x_sig**2)).clip(0,1)*(a_y *T.exp(-0.5*(v)**2/y_sig**2)).clip(0,1)
        super(TiltedGaussKernel2d,self).__init__(kernel,name=name)
    def compute(self):
        f = theano.function([],self.graph)
        return f()

class Gabor(GraphWrapper):
    def __init__(self,x_sig,y_sig,x_offset=0,y_offset=0,freq=1.0,phi=0,phase=0.0,epsilon=0.0005,
                 name='gabor_kernel', even=False,resolution=None):
        """
            A gabor filter is the product of a gaussian and a cosine.           

            `x_sig` and `y_sig` are the width of the gaussian in the two principal directions.

            `phi` is the rotation in rad. when `phi` = 0, x_sig aligns with the x axis,
            y_sig aligns with the y axis.

            `freq` is the frequency of the oscillation in 1/visual degrees.
            The direction of the oscillation is orthogonal to `phi` (along `y_sig`).
            `phase` is the phase of the oscillation from `-pi` to `pi` (or `0` to `2*pi`).
            Default is `0`, ie. the center has value `1.0` and falls off symmetrically.

            The size of the filter is determined automatically and is always square.
            The offsets are from the center, ie. with each increase in x_offset, the filter
            grows by twice that amount.

        """
        if resolution is None:
            resolution = default_resolution
        x_sig = as_parameter(x_sig,name='x_sig')*resolution.var_pixel_per_degree
        y_sig = as_parameter(y_sig,name='y_sig')*resolution.var_pixel_per_degree
        x_offset = as_parameter(x_offset,name='x_offset')
        y_offset = as_parameter(y_offset,name='y_offset')
        phi = as_parameter(phi,name='phi')
        epsilon = as_parameter(epsilon,name='epsilon')
        gabor_freq = as_parameter(freq,name='freq')/resolution.var_pixel_per_degree
        gabor_phase = as_parameter(phase,name='phase')
        a_x = 1.0/(x_sig * T.sqrt(2.0*np.pi))
        x_min = T.ceil(T.sqrt(-2.0*(x_sig**2)*T.log(epsilon/a_x)))
        a_y = 1.0/(y_sig * T.sqrt(2.0*np.pi))
        y_min = T.ceil(T.sqrt(-2.0*(y_sig**2)*T.log(epsilon/a_y)))
        length = T.max([x_min,y_min]) + T.max([abs(x_offset),abs(y_offset)])
        X = T.arange(1.0-length-(0.5 if even else 0.0),length+(0.5 if even else 0.0))
        Y = T.arange(1.0-length-(0.5 if even else 0.0),length+(0.5 if even else 0.0))
        u = T.sin(phi)*(X.dimshuffle(0,'x')+x_offset) + T.cos(phi)*(Y.dimshuffle('x',0)+y_offset)
        v = T.cos(phi)*(X.dimshuffle(0,'x')+x_offset) - T.sin(phi)*(Y.dimshuffle('x',0)+y_offset)
        kernel = (a_x * T.exp(-0.5*(u)**2/x_sig**2)).clip(0,1)*(a_y *T.exp(-0.5*(v)**2/y_sig**2)).clip(0,1)
        kernel = kernel * T.cos(u*gabor_freq + gabor_phase)
        super(Gabor,self).__init__(kernel,name=name)
    def compute(self):
        f = theano.function([],self.graph)
        return f()  

def sum_kernels(kernels):
    max_shape = T.max([k.graph.shape for k in kernels],axis=0)
    new_k = T.zeros((max_shape[0],max_shape[1]),dtype=float)
    for k in kernels:
        x1 = T.floor((max_shape[0] - k.graph.shape[0])/2)
        x2 = x1 + k.graph.shape[0]
        y1 = T.floor((max_shape[1] - k.graph.shape[1])/2)
        y2 = y1 + k.graph.shape[1]
        new_k = T.set_subtensor(new_k[x1:x2,y1:y2], new_k[x1:x2,y1:y2] + k)
    return GraphWrapper(new_k,'Sum')

class FakeFilter(GraphWrapper):
    def __init__(self,list_of_filters,name='resize_convolution_kernel'):
        zeros = T.zeros([1+T.sum([l.shape[i]-1 for l in list_of_filters]) for i in range(list_of_filters[0].ndim)])
        if zeros.ndim == 5:
            one = T.set_subtensor(zeros[:,0,:,zeros.shape[3]//2,zeros.shape[4]//2], 1)
        elif zeros.ndim == 3:
            one = T.set_subtensor(zeros[0,zeros.shape[1]//2,zeros.shape[2]//2], 1)
        elif zeros.ndim == 2:
                    one = T.set_subtensor(zeros[zeros.shape[0]//2,zeros.shape[1]//2], 1)
        elif zeros.ndim == 1:
                    one = T.set_subtensor(zeros[0], 1)
        else:
            raise Exception("Filters do not have 5,3,2 or 1 dimensions!")
        super(FakeFilter,self).__init__(one,name=name)
    def compute(self):
        f = theano.function([],self.graph)
        return f()

class FakeFilter3d(GraphWrapper):
    def __init__(self,list_of_filters,name='resize_convolution_kernel'):
        list_of_filters = [theano_utils.make_nd(l,3) for l in list_of_filters]
        zeros = T.zeros([1+T.sum([l.shape[i]-1 for l in list_of_filters]) for i in range(list_of_filters[0].ndim)])
        if zeros.ndim == 5:
            one = T.set_subtensor(zeros[:,0,:,zeros.shape[3]//2,zeros.shape[4]//2], 1)
        elif zeros.ndim == 3:
            one = T.set_subtensor(zeros[0,zeros.shape[1]//2,zeros.shape[2]//2], 1)
        elif zeros.ndim == 2:
                    one = T.set_subtensor(zeros[zeros.shape[0]//2,zeros.shape[1]//2], 1)
        elif zeros.ndim == 1:
                    one = T.set_subtensor(zeros[0], 1)
        else:
            raise Exception("Filters do not have 5,3,2 or 1 dimensions!")
        super(FakeFilter3d,self).__init__(one,name=name)
    def compute(self):
        f = theano.function([],self.graph)
        return f()

class FakeFilter5d(GraphWrapper):
    def __init__(self,list_of_filters,name='resize_convolution_kernel'):
        list_of_filters = [theano_utils.make_nd(l,5) for l in list_of_filters]
        zeros = T.zeros([1+T.sum([l.shape[i]-1 for l in list_of_filters]) for i in range(list_of_filters[0].ndim)])
        if zeros.ndim == 5:
            one = T.set_subtensor(zeros[:,0,:,zeros.shape[3]//2,zeros.shape[4]//2], 1)
        elif zeros.ndim == 3:
            one = T.set_subtensor(zeros[0,zeros.shape[1]//2,zeros.shape[2]//2], 1)
        elif zeros.ndim == 2:
                    one = T.set_subtensor(zeros[zeros.shape[0]//2,zeros.shape[1]//2], 1)
        elif zeros.ndim == 1:
                    one = T.set_subtensor(zeros[0], 1)
        else:
            raise Exception("Filters do not have 5,3,2 or 1 dimensions!")
        super(FakeFilter5d,self).__init__(one,name=name)
    def compute(self):
        f = theano.function([],self.graph)
        return f()

