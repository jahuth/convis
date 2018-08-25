import numpy as np

def _to_numpy(x):
    # should become obsolete. use _array instead.
    try:
        # assuming a torch Variable on the gpu
        x = x.data.cpu().numpy()
    except:
        x = np.array(x)
    return x   


def _array(t):
    import torch
    from distutils.version import LooseVersion
    if (LooseVersion(torch.__version__) < LooseVersion('0.4.0')) and (type(t) == torch.autograd.variable.Variable):
        return t.data.cpu().numpy()
    else:
        if hasattr(t,'detach'):
            return np.array(t.detach())
        return np.array(t)

def make_tensor_5d(t):
    """Extends a tensor or numpy array to have exactly 5 dimensions.
    
    In doing so, it interprets 1d tensors as being aligned
    in time, 2d tensors to be aligned in space, etc.

        - 1d: t[None,None,:,None,None]
        - 2d: t[None,None,None,:,:]
        - 3d: t[None,None,:,:,:]
        - 4d: t[None,:,:,:,:]
        - 5d: t
        - 6d and more: selects the first element of t until t has 5 dimensions

    """
    if type(t.shape) is tuple:
        # for numpy arrays
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
        return t
    else:
        # for pytorch tensors
        if len(t.size()) > 5:
            while len(t.shape) > 5:
                t = t[0]
        if len(t.size()) == 4:
            t = t[None]
        if len(t.size()) == 3:
            t = t[None,None]
        if len(t.size()) == 2:
            t = t[None,None,None]
        if len(t.size()) == 1:
            t = t[None,None,:,None,None]    
        return t


def extend_to_match(a,b,align=None,mode='linear_ramp',end_values=0.0,dims=5):
    """Extends tensor :attr:`a` to have a shape *at least* as big as `b`.

    If `dims` is equal to 5, :func:`make_tensor_5d` is used
    to add dimensions until both tensors have 5 dimensions.

    By default, the first three dimensions will
    be padded by aligning the smaller tensor at
    the beginning of the larger tensor and the last
    two (spatial) dimensions are aligned centered
    in the larger tensor.

    If `dim` is *not* equal to 5, all dimensions
    are centered.

    Parameters
    ----------

        a (numpy array or tensor):
            the tensor being extended
        b (numpy array or tensor):
            the tensor with (potentially) larger dimensions
        align (list or string):
            if a list: determines the alignment for each dimension
            if a sting: gives the alignment mode for all dimensions
            possible values:  strings starting with : `'c'`, `'b'` and `'e'`
            standing for: `'center'`, `'begin'`, `'end'`
        mode (`str`):
            see `mode` argument of :func:`numpy.pad`
            default: `'linear_ramp'`,
        end_values (`float`):
            see `end_values` argument of :func:`numpy.pad`
            default: `0.0`
        dims (`int`, specifically 5 or anything else):
            if 5, tensors will be extended so that they have exactly 5 dimensions using :func:`make_tensor_5d`


    Examples
    --------

    To make sure both have the same shape:

        >>> a = convis.utils.extend_to_match(a,b)
        >>> b = convis.utils.extend_to_match(b,a)


    """
    if dims == 5:
        a = make_tensor_5d(a)
        b = make_tensor_5d(b)
        if align is None:
            align=['begin','begin','begin','center','center']
    if align is None:
        align = ['center']*len(a.shape)
    if type(align) is str:
        align = [align]*len(a.shape)
    for i in range(len(a.shape)):
        if a.shape[i] < b.shape[i]:
            shp = [(0,0),(0,0),(0,0),(0,0),(0,0)]
            d = b.shape[i] - a.shape[i]
            if align[i].startswith('c'):
                shp[i] = (int(d/2),d-int(d/2))
            elif align[i].startswith('b'):
                shp[i] = (0,d)
            elif align[i].startswith('e'):
                shp[i] = (d,0)
            else:
                raise Exception('Alignment '+str(align[i])+' not undestood! Must be one of (c)enter, (b)egin, (e)nd')
            a = np.pad(a,shp,mode=mode,end_values=end_values)
    return a

def subtract_tensors(a,b,align=None,dims=5):
    """Makes it easy to subtract two 5d tensors from each other, even if they have different shapes!

    If `dims` is equal to 5, :func:`make_tensor_5d` is used
    to add dimensions until both tensors have 5 dimensions.

    By default, the first three dimensions will
    be padded by aligning the smaller tensor at
    the beginning of the larger tensor and the last
    two (spatial) dimensions are aligned centered
    in the larger tensor.

    If `dim` is *not* equal to 5, all dimensions
    are centered.


    Parameters
    ----------

        a (numpy array or tensor):
            the first of the two tensors
        b (numpy array or tensor):
            the second of the two tensors
        align (list or string):
            if a list: determines the alignment for each dimension
            if a sting: gives the alignment mode for all dimensions
            possible values: 
                strings starting with : 'c','b' and 'e'
                standing for: 'center', 'begin', 'end'
        dims (int, specifically 5 or anything else):
            if 5, tensors will be extended so that they have exactly 5 dimensions using :func:`make_tensor_5d`

    """
    if dims == 5:
        a = make_tensor_5d(a)
        b = make_tensor_5d(b)
    a = extend_to_match(a,b,align=align)
    b = extend_to_match(b,a,align=align)
    return a - b       

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
    w = _array(w)
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
    w = _array(w)
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

def plot_tensor(t,n_examples=5,max_lines=16,tlim=None,xlim=None,ylim=None):
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
        tlim (tuple or int):
            time range included in the plot
        xlim (tuple or int):
            x range included in the example frames
        ylim (tuple or int):
            y range included in the example frames

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
    t_0 = 0
    t_end = t.shape[2]
    if tlim is not None:
        if type(tlim) in [tuple,list]:
            t = t[:,:,tlim[0]:tlim[1],:,:]
            t_0 = tlim[0]
            t_end = tlim[1]
        else:
            t = t[:,:,:tlim,:,:]
            t_end = tlim
    if xlim is not None:
        if type(xlim) in [tuple,list]:
            t = t[:,:,:,xlim[0]:xlim[1],:]
        else:
            t = t[:,:,:,:xlim,:]
    if ylim is not None:
        if type(ylim) in [tuple,list]:
            t = t[:,:,:,:,ylim[0]:ylim[1]]
        else:
            t = t[:,:,:,:,:ylim]
    o_mean= np.mean(t,(0,1,3,4))
    o_min= np.min(t,(0,1,3,4))
    o_max= np.max(t,(0,1,3,4))
    ax1 = plt.subplot2grid((2, n_examples), (0, 0), colspan=n_examples)
    plt.fill_between(t_0+np.arange(len(o_min)),o_min,o_max,alpha=0.5)
    plt.plot(t_0+np.arange(len(o_min)),o_mean)
    line_steps = int(min(t.shape[3],t.shape[4])/min(np.sqrt(max_lines),min(t.shape[3],t.shape[4])))
    for i in np.arange(0,t.shape[3],line_steps):
        for j in np.arange(0,t.shape[4],line_steps):
            plt.plot(t_0+np.arange(len(o_min)),t[0,0,:,i,j],'-',color='orange',alpha=0.2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    plt.xlim(t_0,t_end)
    plt.title('Output Tensor')
    for i,_t in enumerate(np.linspace(0,t.shape[2]-1,n_examples)):
        ax2 = plt.subplot2grid((2, n_examples), (1, i))
        plt.imshow(t[0,0,int(_t),:,:],vmin=np.min(t),vmax=np.max(t))
        plt.axis('off')
        plt.title(str(int(t_0+_t)))
    return plt.gcf()


def plot_tensor_with_channels(t,n_examples=7,max_lines=9):
    """Plots a 5d tensor as a line plot (min,max,mean and example timeseries) and a sequence of image plots and respects channels and batches.

        (might replace plot_tensor if it is more reliable and looks nicer)


        .. plot::
            :include-source:

            import convis
            inp = convis.samples.moving_grating(2000,50,50)
            inp = np.concatenate([-1.0*inp[None,None],2.0*inp[None,None],inp[None,None]],axis=0)
            convis.utils.plot_tensor_with_channels(inp)


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
            inp = np.concatenate([-1.0*inp[None,None],2.0*inp[None,None],inp[None,None]],axis=0)
            convis.utils.plot_tensor_with_channels(inp,n_examples=10,max_lines=42)


        .. plot::
            :include-source:

            import convis
            inp = convis.samples.moving_grating(2000,50,50)
            inp = np.concatenate([-1.0*inp[None,None],2.0*inp[None,None],inp[None,None]],axis=1)
            convis.utils.plot_tensor_with_channels(inp[None,None],n_examples=2,max_lines=2)


    """
    import matplotlib.pylab as plt
    from convis import base
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
    num_lines = t.shape[0]*t.shape[1]
    plt.figure(figsize=(9,2+1.2*num_lines))
    ax1 = plt.subplot2grid((1+num_lines, n_examples), (0, 0), colspan=n_examples)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_i = 0
    for a in np.arange(t.shape[0]):
        for b in np.arange(t.shape[1]):
            o_mean = np.mean(t[a,b],(1,2))
            o_min = np.min(t[a,b],(1,2))
            o_max = np.max(t[a,b],(1,2))
            plt.fill_between(np.arange(len(o_min)),o_min,o_max,alpha=0.5,color=colors[colors_i%len(colors)])
            plt.plot(np.arange(len(o_min)),o_mean,color=colors[colors_i%len(colors)])
            line_steps = int(min(t.shape[3],t.shape[4])/min(np.sqrt(max_lines),min(t.shape[3],t.shape[4])))
            for i in np.arange(0,t.shape[3],line_steps):
                for j in np.arange(0,t.shape[4],line_steps):
                    plt.plot(np.arange(len(o_min)),t[a,b,:,i,j],'-',color=colors[colors_i%len(colors)],alpha=0.1)
            colors_i+=1
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    plt.title('Output Tensor')
    for j in np.arange(num_lines):
        for i,_t in enumerate(np.linspace(0,t.shape[2]-1,n_examples)):
            ax2 = plt.subplot2grid((1+num_lines, n_examples), (1+j, i))
            a,b = int(j%t.shape[0]),int(j/t.shape[0])
            plt.imshow(t[a,b,int(_t),:,:],vmin=np.min(t),vmax=np.max(t))
            plt.axis('off')
            if num_lines > 1:
                plt.title('['+str(a)+','+str(b)+',..] '+str(int(_t)))
            else:
                plt.title(str(int(_t)))
    plt.tight_layout()
    return plt.gcf()

def mean_as_float(a):
    a = np.mean(np.array(a))
    try:
        a = a.data[0]
    except:
        pass
    return float(a)
