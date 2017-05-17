"""

This module provides sample kernels and inputs.


"""
import numpy as np

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



def sparse_input(t=2000,x=20,y=20,p=0.01):
    if hasattr(p,'shape') or type(p) is list:
        the_input = np.zeros((len(p),x,y))
        for i in range(len(p)):
            the_input[i,:,:] = np.random.rand(x,y) < p[i]
        return the_input
    else:
        the_input = np.random.rand(t,x,y) < p
        return the_input

def moving_bars(t=2000,x=20,y=20,vt=1.0/200.0,vx=3.0,vy=2.0,p=0.01):
    T,X,Y = np.meshgrid(np.linspace(0.0,t,t),np.linspace(-1.0,1.0,x),np.linspace(-1.0,1.0,y), indexing='ij')
    return np.sin(vt*T+vx*X+vy*Y) 

def gabor_kernel(phi=2.0,size=16,resolution=2.0,f=10.0, sigma_x=1, sigma_y=1):
    vals = np.linspace(-0.5*size/resolution,0.5*size/resolution, size)    
    xgrid, ygrid = np.meshgrid(vals,vals)       
    the_gaussian = np.exp(-(sigma_x*xgrid/2.0)**2-(sigma_y*ygrid/2.0)**2)
    the_sine = np.sin(np.sin(phi) * xgrid/(2.0*np.pi / f) + np.cos(phi) * ygrid/(2.0*np.pi / f))
    the_gabor = the_gaussian * the_sine  
    return the_gabor

def random_checker_stimulus(t=2000,x=20,y=20,checker_size=5,seed=123):
    """
        Creates a random checker flicker of uniformly distributed values
        in a grid of `checker_size`x`checker_size` pixels.
    """
    return (np.random.rand(t,
                             int(x)/checker_size + 1,
                             int(y)/checker_size + 1)
            .repeat(checker_size,axis=1)
            .repeat(checker_size,axis=2)[:,:x,:y])