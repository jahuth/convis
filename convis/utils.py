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


        Examples
        --------

            By default this will generate a very long plot where you can see almost nothing:

            .. plot::
                :include-source:

                import convis
                inp = convis.samples.moving_grating(2000,50,50)
                convis.utils.plot_5d_matshow(inp[None,None])


            By limiting the number of frames, the plot shows the frames next to each other:

            .. plot::
                :include-source:

                import convis
                inp = convis.samples.moving_grating(2000,50,50)
                convis.utils.plot_5d_matshow(inp[None,None,::200])

    """
    import matplotlib.pylab as plt
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


        Examples
        --------


            .. plot::
                :include-source:

                import convis
                inp = convis.samples.moving_grating(2000,50,50)
                convis.utils.plot_5d_time(inp[None,None])

    """
    import matplotlib.pylab as plt
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

def plot(x,mode=None,**kwargs):
    """Plots a tensor/numpy array

        If the array has no spatial extend, it will
        draw only a single line plot. If the array
        has no temporal extend (and no color channels etc.),
        it will plot a single frame.
        Otherwise it will use `plot_tensor`.

        Examples
        --------


        .. plot::
            :include-source:

            import convis
            inp = convis.samples.moving_grating(2000,50,50)
            convis.utils.plot(inp[None,None])


    See Also
    --------
    plot_tensor
    plot_5d_matshow
    plot_5d_time
    """
    import matplotlib.pylab as plt
    try:
        # assuming a torch Variable on the gpu
        x = x.data.cpu().numpy()
    except:
        x = np.array(x)
    if len(x.shape) > 5:
        while len(x.shape) > 5:
            x = x[0]
    if len(x.shape) == 4:
        x = x[None]
    if len(x.shape) == 3:
        x = x[None,None]
    if len(x.shape) == 2:
        x = x[None,None,None]
    if len(x.shape) == 1:
        x = x[None,None,:,None,None]
    shp = x.shape
    if len(shp) == 5:
        if mode=='matshow' or np.prod(shp[:3]) == 1:
            # a single frame
            plt.matshow(x,**kwargs)
        elif mode=='lines' or (np.prod(shp[:2]) == 1 and np.prod(shp[3:]) == 1):
            # a single time line
            plt.plot(x.mean((0,1,3,4)),**kwargs)
        else:
            # a set of images?
            plot_tensor(x,**kwargs)
    else:
        print('x has dimensions:',shp)

def plot_tensor(t,n_examples=5,max_lines=16):
    """Plots a 5d tensor as a line plot (min,max,mean and example timeseries) and a sequence of image plots.


        .. plot::
            :include-source:

            import convis
            inp = convis.samples.moving_grating(2000,50,50)
            convis.utils.plot_tensor(inp[None,None])


        Parameters
        ----------
        t (numpy array or PyTorch tensor):
            The tensor that should be visualized
        n_examples (int):
            how many frames (distributed equally over the
            length of the tensor) should be 
        max_lines (int):
            the maximum number of line plots of exemplary
            timeseries that should be added to the temporal 
            plot. The lines will be distributed roughly equally.
            If the tensor is not square in time (x!=y) or either 
            dimension is too small, it is possible that there 
            will be less than `max_lines` lines.


        Examples
        --------


        .. plot::
            :include-source:

            import convis
            inp = convis.samples.moving_grating(2000,50,50)
            convis.utils.plot_tensor(inp[None,None],n_examples=10,max_lines=42)


        .. plot::
            :include-source:

            import convis
            inp = convis.samples.moving_grating(2000,50,50)
            convis.utils.plot_tensor(inp[None,None],n_examples=2,max_lines=2)


    """
    import matplotlib.pylab as plt
    from . import base
    t = base._array(t)
    if len(t.shape) > 5:
        while len(t.shape) > 5:
            t = t[0]
    if len(t.shape) == 4:
        t = t[None]
    if len(t.shape) == 3:
        t = t[None,None]
    if len(t.shape) == 2:
        t = t[None,None,None]
    if len(t.shape) == 1:
        t = t[None,None,:,None,None]
    o_mean= np.mean(t,(0,1,3,4))
    o_min= np.min(t,(0,1,3,4))
    o_max= np.max(t,(0,1,3,4))
    ax1 = plt.subplot2grid((2, n_examples), (0, 0), colspan=n_examples)
    plt.fill_between(np.arange(len(o_min)),o_min,o_max,alpha=0.5)
    plt.plot(np.arange(len(o_min)),o_mean)
    line_steps = int(min(t.shape[3],t.shape[4])/min(np.sqrt(max_lines),min(t.shape[3],t.shape[4])))
    for i in np.arange(0,t.shape[3],line_steps):
        for j in np.arange(0,t.shape[4],line_steps):
            plt.plot(np.arange(len(o_min)),t[0,0,:,i,j],'-',color='orange',alpha=0.2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    plt.title('Output Tensor')
    for i,_t in enumerate(np.linspace(0,t.shape[2]-1,n_examples)):
        ax2 = plt.subplot2grid((2, n_examples), (1, i))
        plt.imshow(t[0,0,int(_t),:,:],vmin=np.min(t),vmax=np.max(t))
        plt.axis('off')
        plt.title(str(int(_t)))
    return plt.gcf()

def mean_as_float(a):
    a = np.mean(np.array(a))
    try:
        a = a.data[0]
    except:
        pass
    return float(a)
