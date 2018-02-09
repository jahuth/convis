"""

This module provides sample kernels and inputs.


"""
import numpy as np
from .base import prepare_input
from . import models
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
    """
        Creates a moving grating stimulus.
        The actual speed and direction of the gratings is
        a combination of temporal, x and y speed.

        Parameters
        ----------
        t: int
            temporal length of the stimulus
        x: int
            width of the stimulus
        y: int
            height of the stimulus
        vt: float
            speed of gratings
        vx: float
            x speed of gratings
        vy: float
            y speed of gratings
    """
    T,X,Y = np.meshgrid(np.linspace(0.0,t,t),np.linspace(-1.0,1.0,x),np.linspace(-1.0,1.0,y), indexing='ij')
    return np.sin(vt*T+vx*X+vy*Y) 

def moving_bar(t=2000,x=20,y=20,
                      bar_pos = -1.0,
                      bar_direction=0.0,
                      bar_width=1.0,
                      bar_v = 0.01,
                      frames_per_second=None,
                      pixel_per_degree=None,
                      sharp = False,
                      temporal_smoothing=1.0):
    """
        Creates a moving bar stimulus

        Parameters
        ----------
        t: int
            temporal length of the stimulus
        x: int
            width of the stimulus
        y: int
            height of the stimulus
        bar_pos: float
            position of the bar from -1.0 to 1.0
        bar_direction: float
            direction of the bar in rad
        bar_width: float
            width of the bar in pixel
        bar_v: float
            speed of the bar
        sharp: bool
            whether the output is binary or smoothed
        temporal_smoothing: float
            if greater than 0.0 and `sharp=True`,
            smooths the stimulus in time with a
            gaussian kernel with width `temporal_smoothing`.
    """
    bar_width = bar_width/float(x)
    T,X,Y = np.meshgrid(np.linspace(0.0,t,t),np.linspace(-1.0,1.0,x),np.linspace(-1.0,1.0,y), indexing='ij')
    width = 2.0 + 0.5*bar_width
    dist = np.abs((X*np.sin(bar_direction)-Y*np.cos(bar_direction) 
                        - bar_pos + width*((T*bar_v)%2.0-1.0))+0.005)
    if sharp:
        return 1.0*(dist<=bar_width)
    else:
        bar = 1.0*(dist<=bar_width)
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(bar,temporal_smoothing,axis=0)

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

class SampleGenerator(models.LN):
    """
        A Linear-Nonlinear model with a random kernel/weight
        that generates random input/output combinations for
        testing purposes.

        Examples
        --------

        >>> g = convis.samples.SampleGenerator('sparse',kernel_size=(20,5,5),p=0.05)
        >>> x,y = g.generate()
        >>> m = convis.models.LN((50,10,10))
        >>> m.set_optimizer.Adam()
        >>> m.optimize(x,y)

        Parameters
        ----------
        kernel : str or numpy array
                Possible values for `kernel`:
                
                    'random' or 'randn': normal distributed kernel values
                    
                    'rand': random kernel values bounded between 0 and 1
                    
                    'sparse': a kernel with a ratio of (approx).
                              `p` 1s and 1-`p` 0s, randomly assigned.
                    
                    or a 5d or 3d numpy array
        kernel_size : tuple(int,int,int)
                specifies the dimensions of the generated input
        dt : int
                length of chunks when evaluating the model
        p : float
                the ratio of 1s for sparse input

        Attributes
        ----------
        conv : convis.filters.Conv3d
            the convolution operation (including the weight)

        Methods
        -------
        generate(input,size,p)
            generates random input and corresponding output

        See Also
        --------
        convis.samples.generate_sample_data()
    """
    def __init__(self,kernel='random',kernel_size=(10,5,5),dt=100,p=0.1):
        super(SampleGenerator,self).__init__()
        if type(kernel) is str:
            if kernel in ['random', 'randn']:
                self.kernel = np.random.randn(*kernel_size)
            elif kernel is 'rand':
                self.kernel = np.random.rand(*kernel_size)
            elif kernel is 'sparse':
                self.kernel = 1.0*(np.random.rand(*kernel_size)<p)
            else:
                raise Exception('`kernel` parameter not recognized!')
        else:
            self.kernel = kernel
        self.conv.set_weight(self.kernel)
        self.dt = dt
    def generate(self,input='random',size=(2000,20,20), p=0.1):
        """
        Parameters
        ----------
        input : str or numpy array

            Possible values for `input`:

                - 'random' or 'randn': normal distributed input
                - 'rand': random input values bounded between 0 and 1
                - 'sparse': input with a ratio of (approx).
                          `p` 1s and 1-`p` 0s, randomly assigned.
                - or a 3d or 5d numpy array.
            
            If input is anything else, `generate` tries to use it
            as input to the model.

        size : tuple(int,int,int)
                specifies the dimensions of the generated input
        p : float
                the ratio of 1s for sparse input

        Returns
        -------
        x :  PyTorch Tensor 
            the randomly created input input 
        y :  PyTorch Variable
            the corresponding output
        """
        if input in ['random', 'randn']:
            x = np.random.randn(*size)
        elif input is 'rand':
            x = np.random.rand(*size)
        elif input is 'sparse':
            x = 1.0*(np.random.rand(*size) < p)
        else:
            x = input
        y = self.run(x,dt=self.dt)[0]
        return x,y

_g = SampleGenerator()

def generate_sample_data(input='random',size=(2000,20,20)):
    """
        Generates random input and sample output using 
        a `SampleGenerator` with a random kernel.

        Parameters
        ----------
        input : str or numpy array

            Possible values for `input`:

                - 'random' or 'randn': normal distributed input
                - 'rand': random input values bounded between 0 and 1
                - 'sparse': input with a ratio of (approx).
                          `p` 1s and 1-`p` 0s, randomly assigned.
                - or a 3d or 5d numpy array.

        size : tuple(int,int,int)
                specifies the dimensions of the generated input

        Returns
        -------
        x :  PyTorch Tensor 
            the randomly created input input 
        y :  PyTorch Variable
            the corresponding output

        The kernel will be different each time 
        `convis` is (re)loaded, but constant in one
        session. The "secret" `SampleGenerator` is 
        available as `convis.samples._g`.


        Examples
        --------

        >>> m = convis.models.LN((50,10,10))
        >>> m.set_optimizer.Adam()
        >>> x,y = convis.samples.generate_sample_data()
        >>> m.optimize(x,y)

        See Also
        --------
        SampleGenerator
    """
    return _g.generate(input, size)