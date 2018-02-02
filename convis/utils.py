import matplotlib.pylab as plt
import numpy as np

def _to_numpy(x):
    try:
        # assuming a torch Variable on the gpu
        x = x.data.cpu().numpy()
    except:
        x = np.array(x)
    return x   

def plot_5d_matshow(w,border=True, dims=[(0,1,3),(2,4)], border_val = 0.0, **kwargs):
    """
        Plots a 5d Tensor as a concatenation of spatial maps.

        The 5 dimensions have to be separated into two sets which
        will correspond to the two dimensions of the image.


            plot_5d_matshow(w, dims=[(0,3),(2,1,4)])

        Other arguments and keyword arguments are passed
        to `matplotlib.pylab.plot()`

        `border_val` can be a float value or 'max', 'min' or 'mean',
        in which case the corresponding value will be taken from w.
    """
    assert len(dims) == 2, "A 2d plot can only show two dimensions."
    w = _to_numpy(w)
    if len(w.shape) == 3:
        w = w[None,None]
    if border:
        if border_val is 'max':
            border_val = np.max(w)
        if border_val is 'min':
            border_val = np.min(w)
        if border_val is 'mean':
            border_val = np.mean(w)
        w_ = border_val*np.ones((w.shape[0],w.shape[1],w.shape[2],w.shape[3]+1,w.shape[4]+1))
        w_[:,:,:,:-1,:-1] = w
        w = w_
    w_ = np.transpose(w,tuple([i for ii in dims for i in ii])).reshape(*[np.prod([w.shape[d] for d in dim]) for dim in dims])
    plt.matshow(w_, **kwargs)
    return w_

def plot_5d_time(w, lsty='-', mean=tuple(), time=(2,), *args, **kwargs):
    """
        Plots a line plot from a 5d tensor.

        Dimensions in argument `mean` will be combined.
        If `mean` is `True`, all 4 non-time dimensions will be 
        averaged.

        Other arguments and keyword arguments are passed
        to `matplotlib.pylab.plot()`
    """
    w = _to_numpy(w)
    lines=(0,1,2,3,4)
    if mean is True:
        mean = (0,1,3,4)
    lines = [l for l in lines if l not in mean and l not in time]
    dims = [lines,mean,time]
    x = np.transpose(w,tuple([i for ii in dims for i in ii]))
    x = x.reshape(*[int(np.prod([w.shape[d] for d in dim])) for dim in dims]).mean(1).transpose()
    plt.plot(x, lsty, *args, **kwargs)
    return x

def plot(x,**kwargs):
    try:
        # assuming a torch Variable on the gpu
        x = x.data.cpu().numpy()
    except:
        x = np.array(x)
    shp = x.shape
    if len(shp) == 5:
        if np.prod(shp[:3]) == 1:
            # a single frame
            plt.matshow(x,**kwargs)
        elif np.prod(shp[:2]) == 1 and np.prod(shp[3:]) == 1:
            # a single time line
            plt.plot(x.mean((0,1,3,4)),**kwargs)
        else:
            # a set of images?
            plot_5d_matshow(x)
    else:
        print('x has dimensions:',shp)
