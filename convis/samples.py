"""

This module provides sample kernels and inputs.


"""
import numpy as np
from .base import prepare_input
##############################################################################
#
#  Sample Kernels
#

def text_kernel():
    kernel = np.zeros((16,16))
    #con
    con = [list(line) for line in 
           "0011000110010010,0100101001011010,0100001001011010,0100101001010110,0011000110010110".split(',')]
    #vis
    vis = [list(line) for line in 
           "0010001010011100,0010001010110010,0001010010011000,0001010010100110,0000100010011100".split(',')]

    con_x, con_y = np.where(np.array(con) == '1')
    kernel[con_x+2,con_y] = 1.0
    vis_x, vis_y = np.where(np.array(vis) == '1')
    kernel[vis_x+9,vis_y] = 1.0
    return kernel



def gabor_kernel(phi=2.0,size=16,resolution=2.0,f=10.0, phase=0.0, sigma_x=1, sigma_y=1):
    vals = np.linspace(-0.5*size/resolution,0.5*size/resolution, size)    
    xgrid, ygrid = np.meshgrid(vals,vals)       
    the_gaussian = np.exp(-(sigma_x*xgrid/2.0)**2-(sigma_y*ygrid/2.0)**2)
    the_sine = np.sin(np.sin(phi) * xgrid/(2.0*np.pi / f) + np.cos(phi) * ygrid/(2.0*np.pi / f) + phase)
    the_gabor = the_gaussian * the_sine  
    return the_gabor

##############################################################################
#
#  Sample Inputs
#

def sparse_input(t=2000,x=20,y=20,p=0.01,
                            frames_per_second=None,
                            pixel_per_degree=None):
    if hasattr(p,'shape') or type(p) is list:
        the_input = np.zeros((len(p),x,y))
        for i in range(len(p)):
            the_input[i,:,:] = 1.0*(np.random.rand(x,y) < p[i])
        return the_input
    else:
        the_input = 1.0*(np.random.rand(t,x,y) < p)
        return the_input

def moving_grating(t=2000,x=20,y=20,vt=1.0/200.0,vx=3.0,vy=2.0,p=0.01,
                            frames_per_second=None,
                            pixel_per_degree=None):
    T,X,Y = np.meshgrid(np.linspace(0.0,t,t),np.linspace(-1.0,1.0,x),np.linspace(-1.0,1.0,y), indexing='ij')
    return np.sin(vt*T+vx*X+vy*Y) 


def random_checker_stimulus(t=2000,x=20,y=20,checker_size=5,seed=123,
                            frames_per_second=None,
                            pixel_per_degree=None):
    """
        Creates a random checker flicker of uniformly distributed values
        in a grid of `checker_size`x`checker_size` pixels.
    """
    return (np.random.rand(t,
                             int(x)/checker_size + 1,
                             int(y)/checker_size + 1)
            .repeat(checker_size,axis=1)
            .repeat(checker_size,axis=2)[:,:x,:y])

def pulse(t=2000,x=20,y=20, pulse_length=100,
                            frames_per_second=None,
                            pixel_per_degree=None):
    data = np.zeros((t, x, y))
    data[np.arange(t)%(2*pulse_length) < pulse_length,:,:] = 1.0
    return data

def chirp(t=2000,x=20,y=20, pulse_on=500, pulse_off=1000,
                            amp_on=1500, amp_off=2500,
                            amp_freq = 5.0,
                            amp_1=0.0,amp_2=1.0,
                            freq_on=3000, freq_off=4000,
                            freq_1=0.1,freq_2=10.0,
                            stimulus_off = 4500,
                            scale=True,
                            frames_per_second=1000.0,
                            pixel_per_degree=None):
    if scale:
        total_t = stimulus_off #pulse_on + pulse_off + amp_on + amp_off + freq_on + freq_off
        total_scale = float(t)/float(total_t)
        pulse_on = int(pulse_on * total_scale)
        pulse_off = int(pulse_off * total_scale)
        amp_on = int(amp_on * total_scale)
        amp_off = int(amp_off * total_scale)
        freq_on = int(freq_on * total_scale)
        freq_off = int(freq_off * total_scale)
        stimulus_off = int(stimulus_off * total_scale)
    data = np.zeros((t, x, y))
    data[pulse_on:pulse_off,:,:] = 1.0
    one = np.ones((1, x, y))
    data[amp_on:amp_off,:,:] = one*(np.linspace(amp_1,amp_2,amp_off-amp_on)
                                    *np.sin(2.0*np.pi*np.arange(amp_off-amp_on)*amp_freq/frames_per_second))[:,None,None]
    data[freq_on:freq_off,:,:] = one*np.sin(2.0*np.pi*np.linspace(freq_1,freq_2,freq_off-freq_on)
                                            *np.arange(freq_off-freq_on)/frames_per_second)[:,None,None]
    return data


class StimulusSize(object):
    """
        This class holds information about sample stimulus size.        
    """
    def __init__(self,t=2000,x=20,y=20, pixel_per_degree=10, frames_per_second=1000.0, cuda=False, prepare=False):
        self.t = int(t)
        self.x = int(x)
        self.y = int(y)
        self.frames_per_second = frames_per_second
        self.pixel_per_degree = pixel_per_degree
        self._stimulus_functions = {
                'sparse_input': sparse_input,
                'moving_grating': moving_grating,
                'pulse':pulse,
                'chirp':chirp
            }
        if cuda:
            prepare = True
        self._cuda = cuda
        self._prepare = prepare
        self.__dict__.update(self._stimulus_functions)
    def __getattr__(self,k):
        if self._prepare:
            if k in self._stimulus_functions.keys():
                return lambda *args, **kwargs: prepare_input(self._stimulus_functions[k](t=self.t,x=self.x,y=self.y,pixel_per_degree=self.pixel_per_degree,frames_per_second=self.frames_per_second,*args,**kwargs), cuda=self._cuda)
        else:            
            if k in self._stimulus_functions.keys():
                return lambda *args, **kwargs: self._stimulus_functions[k](t=self.t,x=self.x,y=self.y,pixel_per_degree=self.pixel_per_degree,frames_per_second=self.frames_per_second,*args,**kwargs)

default_stimulus_size = StimulusSize()
default_stimulus_size_cuda = StimulusSize(cuda = True)
cuda = StimulusSize(cuda = True)