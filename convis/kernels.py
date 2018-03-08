
"""
                                                                             
Kernels
-------

This module collects convolution kernels.

"""

from .numerical_filters import exponential_filter_1d, exponential_filter_5d
from .numerical_filters import exponential_highpass_filter_1d, exponential_highpass_filter_5d
from .numerical_filters import gauss_filter_2d, gauss_filter_3d, gauss_filter_5d
from .samples import text_kernel, gabor_kernel
import torch

class ExponentialKernel(torch.nn.Module):
    """Derivable Exponential Filter Cascade with fixed length
    
    .. note::
        `n` is a non-derivable parameter,
        two exponential filters with different
        `n` might never converge to the same `tau`
        when fitted.
    
    """
    def __init__(self, tau = 0.01, n=0, amplification=1.0, length=100):
        super(ExponentialKernel, self).__init__()
        self.length = length
        self.tau = Parameter(tau)
        self.n = n#torch.autograd.Variable(torch.Tensor(n))
        self.amplification = Parameter(amplification)
    def forward(self):
        tau_in_steps = self.tau*default_resolution.steps_per_second
        if self.n == 0:
            a = self.amplification/tau_in_steps
            t = torch.linspace(1.0,self.length,int(self.length))
            x = torch.autograd.Variable(torch.linspace(0.4,self.length-0.6,int(self.length)))
            kernel = torch.exp(-x/tau_in_steps)
            kernel = kernel * a
        else:
            a = self.amplification
            length = (int(-tau_in_steps*torch.log(default_resolution.filter_epsilon/a))-self.n)
            t = torch.autograd.Variable(torch.linspace(1.0,self.n*self.length,int(self.n*self.length)))
            kernel = self.amplification * (self.n*t)**self.n * torch.exp(-self.n*t/tau_in_steps) / (np.math.factorial(self.n-1) * tau_in_steps**(self.n+1))
        return kernel
