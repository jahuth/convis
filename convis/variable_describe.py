"""
 * Describing Variables: :meth:`describe`
 * Aniamting Tensors: :meth:`animate`, :meth:`animate_to_video`
 * Plotting Tensors: :meth:`plot_3d_tensor_as_3d_plot`
"""

from types import ModuleType, FunctionType
import re
import numpy as np
from .o import save_name
#from .variables import get_convis_attribute, has_convis_attribute, full_path
import inspect
from future.utils import iteritems as _iteritems
import torch

def has_convis_attribute(o,k):
    return hasattr(o,k)
def get_convis_attribute(o,k,default=None):
    return getattr(o,k,default)

plotting_possible = False
plotting_exceptions = []
do_3d_plot = True
try:
    import matplotlib
    import matplotlib.pylab as plt
    from IPython.core import pylabtools as pt
    gui, backend = pt.find_gui_and_backend('inline', 'inline')
    from IPython.core.pylabtools import activate_matplotlib
    activate_matplotlib(backend)
    plotting_possible = True
except Exception as e:
    plotting_exceptions.append(e)
    pass


doc_urls = [
    ('convis.filters.retina', 'https://jahuth.github.io/convis/docs_retina.html#'),
    ('convis.filters', 'https://jahuth.github.io/convis/docs_filters.html#'),
    ('convis.models', 'https://jahuth.github.io/convis/docs_models.html#'),
    ('convis.retina', 'https://jahuth.github.io/convis/docs_retina.html#'),
    ('convis.base','https://jahuth.github.io/convis/docs.html#'),
    ('convis.streams','https://jahuth.github.io/convis/docs_streams.html#'),
    ('convis.samples','https://jahuth.github.io/convis/docs.html#')
]

def describe(v,**kwargs):
    """All-purpose function to describe many convis objects

    The output adapts itself to being displayed in a
    notebook or text console.
    """
    return _Descriptor(v,**kwargs)


def help(v,**kwargs):
    return _Descriptor({'name': get_convis_attribute(v,'name'),
                        'doc': get_convis_attribute(v,'doc')},**kwargs)

class _Descriptor(object):
    def __init__(self,v,**kwargs):
        self.v = v
        self.kwargs = kwargs
    def __repr__(self):
        return repr(describe_dict(self.v))
    def __str__(self):
        return describe_text(self.v)
    def _repr_pretty_(self,  p, cycle):
        p.text(describe_text(self.v))
    def _repr_html_(self):
        return describe_html(self.v,wrap_in_html=False,**self.kwargs)

def describe_text(v, indent=' '):
    pure_indent = len(indent) * ' '
    def repr_if_not_str(s):
        if type(s) is str:
            return s
        return re.sub( '\n+', '\n', repr(s) )
    if hasattr(v,'__iteritems__'):
        return (indent + ' keys:' + ', '.join([k for k,vv in v.__iteritems__()]) + '\n' +
                '\n'.join([indent[:-1] + '---\\ ' + k + '\n'+describe_text(vv,indent=pure_indent+'  | ')+'\n' for k,vv in v.__iteritems__()]))
    if type(v) in [list, tuple] or hasattr(v,'__iter__'):
        try:
            return '\n'.join([describe_text(vv,indent=indent+'    ') for vv in v])
        except:
            # Tensor Variables love to raise TypeErrors when iterated over
            pass
    d = ''
    if isinstance(v, ModuleType):
        d += str(getattr(v,'__doc__',""))
        for f in dir(v):
            vv = getattr(v,f)
            d += f +  '\n' + '-'*len(f) + '\n'
            d += str(getattr(vv,'__doc__',str(type(vv))))
        return ('\n'.join([indent + line for line in d.split('\n')])).replace('\n\n','\n')
    for k in ['name','simple_name','doc','config_key','optimizable','node','save','init','get','set','auto_name']:
        if has_convis_attribute(v,k):
            d+= str(k) +':'+ repr_if_not_str(get_convis_attribute(v,k)) + '\n'
    try:
        d+= 'value:' + repr_if_not_str(v.get_value()) + '\n'
    except:
        pass
    try:
        d+= 'got: ' + repr_if_not_str(get_convis_attribute(v,'get',(tu.create_context_O(v)))) + '\n'
    except:
        pass
    return ('\n'.join([indent + line for line in d.split('\n')])).replace('\n\n','\n')

def describe_dict(v):
    if type(v) in [list, tuple] or hasattr(v,'__iter__'):
        try:
            return [describe_dict(vv) for vv in v]
        except:
            # Tensor Variables love to raise TypeErrors when iterated over
            pass
    d = {}
    if isinstance(v, ModuleType):
        d['doc'] = str(getattr(v,'__doc__',""))
        for f in dir(v):
            vv = getattr(v,f)
            d[f] = str(getattr(vv,'__doc__',str(type(vv))))
        return d
    for k in ['name','simple_name','doc','config_key','optimizable','node','save','init','get','set','auto_name']:
        if has_convis_attribute(v,k):
            d[k] = get_convis_attribute(v,k)
    try:
        d['value'] = v.get_value()
    except:
        pass
    try:
        d['got'] = get_convis_attribute(v,'get',(tu.create_context_O(v)))
    except:
        pass
    return d

def _plot_to_string():
    import StringIO, urllib
    import base64
    import matplotlib.pylab as plt
    imgdata = StringIO.StringIO()
    plt.savefig(imgdata)
    plt.close()
    imgdata.seek(0) 
    image = base64.encodestring(imgdata.buf)  
    return str(urllib.quote(image))    

def _tensor_to_html(t,title='',figsize=(5,4),line_figsize=(5,1.5),line_kwargs={},imshow_kwargs={},preamble=True,**other_kwargs):
    """
        This function plots/prints numerical objects of 0,1,2,3 and 5 dimensions such that it can be displayed as html.
    """
    kwargs = {'title':title,'figsize':figsize,'line_figsize':line_figsize,'line_kwargs':line_kwargs,'imshow_kwargs':imshow_kwargs}
    imshow_kwargs['interpolation'] = imshow_kwargs.get('interpolation','nearest')
    if type(t) == int or type(t) == float:
        return str(t)
    elif type(t) == np.ndarray:
        if plotting_possible:
            import matplotlib.pylab as plt
            if len(t.shape) == 0:
                return str(t)
            if len(t.shape) == 1:
                if t.shape[0] == 1:
                    if preamble is False:
                        return str(t[0])
                    return str(t[0]) + ' (1,)'
                else:
                    plt.figure(figsize=line_figsize)
                    if title != '':
                        plt.title(title)
                    plt.plot(t,**line_kwargs)
                    plt.tight_layout()
                    if preamble is False:
                        return "<img src='data:image/png;base64," + _plot_to_string() + "'>"
                    return "Numpy array "+str(t.shape)+"<br/><img src='data:image/png;base64," + _plot_to_string() + "'>"
            elif len(t.shape) == 2:
                if t.shape[0] == 1 and t.shape[1] == 1:
                    if preamble is False:
                        return str(t[0])
                    return str(t[0]) + ' (1,1)'
                else:
                    if np.abs(float(t.shape[0] - t.shape[1]))/(t.shape[0]+t.shape[1]) < 0.143:
                        # for roughly square 2d objects
                        plt.figure(figsize=figsize)
                        if title != '':
                            plt.title(title)
                        plt.imshow(t,**imshow_kwargs)
                        plt.colorbar()
                    else:
                        plt.figure(figsize=line_figsize)
                        # for 2d objects with one long side:
                        if t.shape[0] > t.shape[1]:
                            plt.plot(t,**line_kwargs)
                        else:
                            plt.plot(t.transpose(),**line_kwargs)
                    plt.tight_layout()
                    if preamble is False:
                        return "<img src='data:image/png;base64," + _plot_to_string() + "'>"
                    return "Numpy array "+str(t.shape)+"<br/><img src='data:image/png;base64," + _plot_to_string() + "'>"
            elif len(t.shape) == 3:
                if t.shape[0] == 1 and t.shape[1] == 1 and t.shape[2] == 1:
                    return str(t[0]) + ' (1,1,1)'
                else:
                    legend = ""
                    if t.shape[0] == 1:
                        img = _tensor_to_html(t[0,:,:],preamble=False,**kwargs)
                    elif t.shape[1] == 1 and t.shape[2] == 1:
                        img = _tensor_to_html(t[:,0,0],preamble=False,**kwargs)
                    else:
                        ## Todo: Plotting profiles of other dimensions iff they show interesting changes
                        #
                        # Right now the profile is plotted if the tensor is sufficiently small.
                        # But a better way would be to determine if the profile is "interesting"
                        # and then sampling a few lines from there.
                        plt.figure(figsize=figsize)
                        if title != '':
                            plt.suptitle(title)
                        plt.subplot(221)
                        plt.title('mean over time')
                        plt.imshow(t.mean(0),**imshow_kwargs)
                        plt.subplot(222)
                        plt.title('mean over x')
                        if t.shape[2] <= 30:
                            plt.plot(t.mean(0),range(t.shape[1]),'k',alpha=0.2,**line_kwargs)
                            legend += 'Grey lines show the profile of the x dimension. '
                        if t.shape[0] <= 100:
                            plt.plot(t.mean(2).transpose(),range(t.shape[1]),'g',alpha=0.2,**line_kwargs)
                        plt.plot(t.mean((0,2)),range(t.shape[1]),**line_kwargs)
                        plt.gca().invert_yaxis()
                        plt.ylabel('y')
                        plt.subplot(223)
                        plt.title('mean over y')
                        if t.shape[1] <= 30:
                            plt.plot(t.mean(0).transpose(),'r',alpha=0.2,**line_kwargs)
                            legend += 'Red lines show the profile of the y dimension. '
                        if t.shape[0] <= 100:
                            plt.plot(t.mean(1).transpose(),'g',alpha=0.2,**line_kwargs)
                            legend += 'Green lines show different time points. '
                        plt.plot(t.mean((0,1)),**line_kwargs)
                        plt.xlabel('x')
                        plt.subplot(224)
                        plt.title('mean over x and y')
                        if t.shape[1]+t.shape[2] <= 2*30:
                            plt.plot(t.reshape((t.shape[0],-1)),color='orange',alpha=0.1,**line_kwargs)
                            legend += 'Orange lines show the spatial profile over time. '
                        plt.plot(t.mean((1,2)),**line_kwargs)  
                        plt.xlabel('time')
                        plt.tight_layout()
                        img = "<img src='data:image/png;base64," + _plot_to_string() + "'>"

                        if do_3d_plot and t.shape[0] <= 20:
                            # TODO: create config for these cutoff values
                            with OrthographicWrapper():
                                ax = plot_3d_tensor_as_3d_plot(t)
                                plt.tight_layout()
                                img += "<img src='data:image/png;base64," + _plot_to_string() + "'>"
                    if preamble is False:
                        return img
                    return "Numpy array "+str(t.shape)+"<br/>"+img+legend            
            elif len(t.shape) == 5:
                # we assume its actually 3d with extra dimensions
                if t.shape[0] == 1 and t.shape[2] == 1:
                    return "Numpy array "+str(t.shape)+"<br/>"+_tensor_to_html(t[0,:,0,:,:],preamble=False,**kwargs)
                else:
                    if t.shape[0] == 1:
                        s = "Numpy array "+str(t.shape)+"<br/>"
                        for i in range(t.shape[2]):
                           s+= _tensor_to_html(t[0,:,i,:,:],preamble=False,**kwargs) + "<br/>"
                        return s
                    elif t.shape[2] == 1:
                        s = "Numpy array "+str(t.shape)+"<br/>"
                        for i in range(t.shape[0]):
                           s+= _tensor_to_html(t[i,:,0,:,:],preamble=False,**kwargs) + "<br/>"
                        return s
                    else:
                        return '5D tensor with too large first or third dimensions!'
        return 'Numpy Array ('+str(t.shape)+')'
    else:
        if preamble is False:
            return str(t[0])
        return str(type(t))+': '+str(t)

on_click_toggle =  """onclick='$(this).parent().children(".description_content").toggle();$(this).parent().children(".description_content_replacer").toggle();'"""

var_name_counter = 0


def animate(ar,skip=10,interval=100):
    """animates a 3d or 5d array in a jupyter notebook

    Returns a `matplotlib` animation object.

    Parameters
    ----------
    ar (np.array):
        3d or 5d array to animate
    skip (int):
        the animation skips this many timesteps
        between two frames.
        When generating an html plot or video for long
        sequences, this should be set to a higher value
        to keep the video short
    interval (int):
        number of milliseconds between two animation
        frames

    Examples
    --------

    To use in a jupyter notebook, use a suitable matplotlib backend::

        >>> %matplotlib notebook
        >>> import convis
        >>> inp = convis.samples.moving_grating(5000)
        >>> convis.animate(inp)
        <matplotlib.animation.FuncAnimation at 0x7f99d1d88750>

    To get a html embedded video, use `convis.variable_describe.animate_to_video`

        >>> %matplotlib notebook
        >>> import convis
        >>> inp = convis.samples.moving_grating(5000)
        >>> convis.variable_describe.animate_to_video(inp)
        <HTML video embedded in the notebook>

    See Also
    --------
    convis.variable_describe.animate_double_plot
    convis.variable_describe.animate_to_video
    convis.variable_describe.animate_to_html
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation 
    import numpy as np
        
    assert len(ar.shape) in [3,5], """The input array has to have 3 or 5 dimensions, not %i!
        Input dimensions are: %s"""%(len(ar.shape),str(ar.shape))

    if len(ar.shape) == 5:
            ar = np.concatenate(np.concatenate(ar,axis=-1),axis=-2)

    skipped_array = ar[::skip,:,:]
    fig, ax = plt.subplots()
    ims = plt.imshow(skipped_array[0],vmin=ar.min(),vmax=ar.max())
    plt.colorbar()
    plt.title(str(0))
    def update(i):
        ims.set_array(skipped_array[i])
        plt.title(str(i*skip))
        return ims
    anim = FuncAnimation(fig, update, frames=len(skipped_array), repeat=True, interval=interval)
    return anim

def animate_double_plot(ar,skip=10,interval=200,window_length = 200):
    """animates two plots to show a 3d or 5d array: a spatial and a temporal scrolling line plot

    5d arrays will be converted into 3d arrays by concatenating
    the batch and channel dimensions in the x and y spatial dimensions.

    Parameters
    ----------
    ar (np.array):
        3d or 5d array to animate
    skip (int):
        the animation skips this many timesteps
        between two frames.
        When generating an html plot or video for long
        sequences, this should be set to a higher value
        to keep the video short
    interval (int):
        number of milliseconds between two animation
        frames
    window_length (int):
        length of the window displayed in the 
        scrolling line plot

    See Also
    --------
    convis.variable_describe.animate
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation 
    import numpy as np
    import convis

    assert len(ar.shape) in [3,5], """The input array has to have 3 or 5 dimensions, not %i!
        Input dimensions are: %s"""%(len(ar.shape),str(ar.shape))

    if len(ar.shape) == 5:
            ar = np.concatenate(np.concatenate(ar,axis=-1),axis=-2)

    skipped_array = ar[::skip,:,:]
    fig, (ax1,ax2) = plt.subplots(1,2)
    plt.subplot(ax1)
    ims = plt.imshow(skipped_array[0],vmin=ar.min(),vmax=ar.max())
    plt_x = np.linspace(0,window_length,window_length)
    plt_mean = ax2.plot(ar[:,:,:].mean((1,2)))
    plt_examples = ax2.plot(ar[:,int(ar.shape[1]/2),int(ar.shape[2]/2)],alpha=0.5,color='black')
    ar_mean = ar[:,:,:].mean((1,2))
    ar_min  = ar[:,:,:].min((1,2))
    ar_max  = ar[:,:,:].max((1,2))
    plt_fill = ax2.fill_between(np.arange(len(ar_min)),ar_min,ar_max,alpha=0.5)
    ar_std_min  = ar_mean - ar[:,:,:].std((1,2))
    ar_std_max  = ar_mean + ar[:,:,:].std((1,2))
    plt_pointer = ax2.axvline(0,color='red')
    plt_fill = ax2.fill_between(np.arange(len(ar_min)),ar_std_min,ar_std_max,alpha=0.5)
    ax2.set_xlim(-0.5*window_length,0.5*window_length)
    plt.colorbar()
    plt.title(str(0))
    plt.tight_layout()
    def update(i):
        ims.set_array(skipped_array[i])
        plt.title(str(i*skip))
        ax2.set_xlim(i*skip-0.5*window_length,(i*skip+0.5*window_length))
        plt_pointer.set_data([i*skip,i*skip],[-100,100])
        return ims,plt_mean
    anim = FuncAnimation(fig, update, frames=len(skipped_array), repeat=True, interval=interval)
    return anim

def animate_to_video(ar,skip=10,interval=100,scrolling_plot=False,window_length=200):
    """animates a 3d or 5d array in a jupyter notebook

    Returns a Jupyter HTML object containing an embedded video
    that can be downloaded.

    Parameters
    ----------
    ar (np.array):
        3d or 5d array to animate
    skip (int):
        the animation skips this many timesteps
        between two frames.
        When generating an html plot or video for long
        sequences, this should be set to a higher value
        to keep the video short
    interval (int):
        number of milliseconds between two animation
        frames
    scrolling_plot (bool):
        whether to plot the spatial and temporal plots
        or only the spatial animation
    window_length (int):
        if `scrolling_plot` is `True`, specifies
        the length of the time window displayed

    Examples
    --------

        >>> %matplotlib notebook
        >>> import convis
        >>> inp = convis.samples.moving_grating(5000)
        >>> convis.variable_describe.animate_to_video(inp)
        <HTML video embedded in the notebook>

    See Also
    --------
    convis.variable_describe.animate
    """
    if scrolling_plot:
        anim = animate_double_plot(ar,skip=skip,interval=interval,window_length=window_length)
    else:
        anim = animate(ar,skip=skip,interval=interval)
    from IPython.display import HTML
    return HTML(anim.to_html5_video())

def animate_to_html(ar,skip=10,interval=100,scrolling_plot=False,window_length=200):
    """animates a 3d or 5d array in a jupyter notebook

    Returns a Jupyter HTML object containing an embedded 
    javascript animated plot.

    Parameters
    ----------
    ar (np.array):
        3d or 5d array to animate
    skip (int):
        the animation skips this many timesteps
        between two frames.
        When generating an html plot or video for long
        sequences, this should be set to a higher value
        to keep the video short
    interval (int):
        number of milliseconds between two animation
        frames
    scrolling_plot (bool):
        whether to plot the spatial and temporal plots
        or only the spatial animation
    window_length (int):
        if `scrolling_plot` is `True`, specifies
        the length of the time window displayed

    Examples
    --------

        >>> %matplotlib notebook
        >>> import convis
        >>> inp = convis.samples.moving_grating(5000)
        >>> convis.variable_describe.animate_to_html(inp)
        <HTML javascript plot embedded in the notebook>

    See Also
    --------
    convis.variable_describe.animate
    """
    if scrolling_plot:
        anim = animate_double_plot(ar,skip=skip,interval=interval,window_length=window_length)
    else:
        anim = animate(ar,skip=skip,interval=interval)
    from IPython.display import HTML
    return HTML(anim.to_jshtml())


def describe_html(v,wrap_in_html=True,**kwargs):
    from IPython.display import HTML
    import uuid
    try:
        import html
    except:
        import cgi as html # fallback escape function
    ##       
    # Handeling other datatypes
    #
    if type(v) == int or type(v) == float:
        s = str(v)
        if not wrap_in_html:
            return s
        return HTML(s)
    elif hasattr(v,'__array__'):
        s = _tensor_to_html(v.__array__(),**kwargs)
        if not wrap_in_html:
            return s
        return HTML(s)
    if isinstance(v, torch.nn.Module):
        return describe_layer_with_html(v, 4, wrap_in_html)
    if isinstance(v, ModuleType):
        uid = uuid.uuid4().hex
        s = """<div class='convis_description module'><b """+on_click_toggle+""">"""+getattr(v,'__name__','(nameless module)')+"""</b>"""
        s += "<div class='description_content_replacer' style='border-left: 2px solid #eee; padding-left: 5px; margin-bottom: 10px; display: none;'>(&#8230;)</div>"
        s += "<div class='description_content' style='border-left: 2px solid #eee; border-top: 2px solid #f8f8f8;  padding-left: 5px; margin-bottom: 10px;  margin-top: 2px;'>"
        s += '<pre>'+str(getattr(v,'__doc__',"(no doc string found)"))+'</pre>'
        for f in dir(v):
            if f.startswith('_'):
                continue
            vv = getattr(v,f)
            if isinstance(vv, ModuleType):
                s += "<div class='convis_description dict_item'><b>"+str(f)+"</b> (module "+str(getattr(vv,'__name__',''))+")</div>"
                continue
            s += "<div class='convis_description dict_item'><b id="+uid+save_name(f)+" "+on_click_toggle+" >"+str(f)+"</b> <a style=\"text-decoration: none;\" href='#"+uid+"''>&#8617;</a>"
            s += "<div class='description_content_replacer' style='border-left: 0px solid #ddd; padding-left: 5px; display: none;'>(&#8230;)</div>"
            s += "<div class='description_content' style='border-left: 0px solid #ddd; padding-left: 5px;'>"
            s += describe_html(vv,wrap_in_html=False,**kwargs)
            s += "</div>"
            s += "</div>"
        s += """</div>"""
        s += """</div>"""
        if not wrap_in_html:
            return s
        return HTML(s)
    if isinstance(v, FunctionType):
        s = """<div class='convis_description module'><b """+on_click_toggle+""">"""+getattr(v,'__name__','(nameless function)')+"""</b>"""
        s += "<div class='description_content_replacer' style='border-left: 2px solid #eee; padding-left: 5px; margin-bottom: 10px; display: none;'>(&#8230;)</div>"
        s += "<div class='description_content' style='border-left: 2px solid #eee; border-top: 2px solid #f8f8f8;  padding-left: 5px; margin-bottom: 10px;  margin-top: 2px;'>"
        s += '<pre>'+str(getattr(v,'__doc__',"(no doc string found)"))+'</pre>'
        s += """</div>"""
        s += """</div>"""
        if not wrap_in_html:
            return s
        return HTML(s)
    if inspect.isclass(v):
        s = """<div class='convis_description module'><b """+on_click_toggle+""">"""+getattr(v,'__name__','(nameless class)')+"""</b>"""
        s += "<div class='description_content_replacer' style='border-left: 2px solid #eee; padding-left: 5px; margin-bottom: 10px; display: none;'>(&#8230;)</div>"
        s += "<div class='description_content' style='border-left: 2px solid #eee; border-top: 2px solid #f8f8f8;  padding-left: 5px; margin-bottom: 10px;  margin-top: 2px;'>"
        s += '<pre>'+str(getattr(v,'__doc__',"(no doc string found)"))+'</pre>'
        s += """</div>"""
        s += """</div>"""
        if not wrap_in_html:
            return s
        return HTML(s)
    if type(v) in [dict] or hasattr(v,'__iteritems__'):
        uid = uuid.uuid4().hex
        s = "<div class='convis_description list'>"
        iteration = v.__iteritems__() if hasattr(v,'__iteritems__') else _iteritems(v)
        s += "<b id="+uid+" "+on_click_toggle+" >+</b>&nbsp;"
        for (k,vv) in iteration:
            s += '| <a style="text-decoration: none; font-size: 8pt;" href="#'+uid+save_name(k)+'">'+str(k)+'</a> '
        s += "<div class='description_content_replacer' style='border-left: 4px solid #f0f0f0; border-top: 4px solid #f8f8f8; padding-left: 10px; margin-bottom: 10px; display: none;'>(&#8230;)</div>"
        s += "<div class='description_content' style='border-left: 4px solid #f0f0f0; border-top: 4px solid #f8f8f8; padding-left: 10px; margin-bottom: 10px;'>"
        iteration = v.__iteritems__() if hasattr(v,'__iteritems__') else _iteritems(v)
        for (k,vv) in iteration:
            s += "<div class='convis_description dict_item'><b id="+uid+save_name(k)+" "+on_click_toggle+" >"+str(k)+"</b> <a style=\"text-decoration: none;\" href='#"+uid+"''>&#8617;</a>"
            s += "<div class='description_content_replacer' style='border-left: 0px solid #ddd; padding-left: 5px; display: none;'>(&#8230;)</div>"
            s += "<div class='description_content' style='border-left: 0px solid #ddd; padding-left: 5px;'>"
            s += describe_html(vv,wrap_in_html=False,**kwargs)
            s += "</div>"
            s += "</div>"
        s += "</div>"
        s += "</div>"
        if not wrap_in_html:
            return s
        return HTML(s)
    if type(v) in [list, tuple] or hasattr(v,'__iter__'):
        try:
            s = "<div class='convis_description list'><b "+on_click_toggle+">List ("+str(len(v))+"):</b>"
            s += "<div class='description_content_replacer' style='border-left: 4px solid #f0f0f0; border-top: 4px solid #f8f8f8; padding-left: 10px; margin-bottom: 10px; display: none;'>(&#8230;)</div>"
            s += "<div class='description_content' style='border-left: 4px solid #f0f0f0; border-top: 4px solid #f8f8f8; padding-left: 10px; margin-bottom: 10px;'>"
            s += '\n'.join([describe_html(vv,wrap_in_html=False,**kwargs) for vv in v])
            s += "</div>"
            s += "</div>"
            if not wrap_in_html:
                return s
            return HTML(s)
        except:
            # Tensor Variables love to raise TypeErrors when iterated over
            pass

    ##       
    # Assuming its a annotated theano variable:
    #
    d = {}
    for k in ['name','simple_name','doc','config_key','optimizable','node','save','init','get','set','variable_type','auto_name']:
        if has_convis_attribute(v,k):
            d[k] = get_convis_attribute(v,k)
    name = d.get('name','') # optional: None handling
    if not type(name) is str or name is '':
        name= repr(v)
    if has_convis_attribute(v,'html_name'):
        name+=' '+str(get_convis_attribute(v,'html_name'))
    #simple_name = str(d.get('simple_name',''))
    s = """<div class='convis_description variable'><b """+on_click_toggle+""">"""+name+"""</b> <small>"""+d.get('variable_type','')+"""</small>"""
    # default: show everything, hide on click;
    s += "<div class='description_content_replacer' style='border-left: 2px solid #eee; padding-left: 5px; margin-bottom: 10px; display: none;'>(&#8230;)</div>"
    s += "<div class='description_content' style='border-left: 2px solid #eee; border-top: 2px solid #f8f8f8;  padding-left: 5px; margin-bottom: 10px;  margin-top: 2px;'>"
    if has_convis_attribute(v,'path'):
        s += "<small>" + full_path(v) + "</small><br/>"
    if has_convis_attribute(v,'doc') and get_convis_attribute(v,'doc') != '':
        s += '<p class="doc" style="padding:2px;">'+get_convis_attribute(v,'doc')+'</p>'
    if has_convis_attribute(v,'owner'):
        s += "<tt style='color: gray;'><small>" + str(v.owner) + "</small></tt><br/>"
    for k in ['auto_name','config_key','optimizable','node','save','init','get','set','state_out_state','param_init','state_init','state_in_state','copied_from','config_key','config_default']:
        if has_convis_attribute(v,k):
            s+= '<div><b>'+str(k)+'</b>: <tt>'+html.escape(str(get_convis_attribute(v,k)))+'</tt></div>'
    try:
        if hasattr(v,'get_value'):
            s+= '<b>value</b>: ' + str(_tensor_to_html(v.get_value(),title=name,**kwargs))
    except Exception as e:
        s+= '<b>value</b>: ' + str(e)
        pass
    try:
        s+= '<b>got</b>: ' + _tensor_to_html(get_convis_attribute(v,'get')(tu.create_context_O(v)),title=name,**kwargs)
    except:
        pass
    s += """</div>"""
    s += """</div>"""
    if not wrap_in_html:
        return s
    return HTML(s)

class OrthographicWrapper():
    """
        This context manager overwrites the persp_transformation function of proj3d
        to perform orthographic projections.
        Plots that are show()n or save()d in this context will use the projection.

        After the context closes, the old projection is restored.

        Examples
        --------

            >>> with convis.variable_describe.OrthographicWrapper():
             ..     plot_3d_tensor_as_3d_plot(ar) # orthographic projection
            >>> plot_3d_tensor_as_3d_plot(ar) # returns to default projection
            

            >>> orth = convis.variable_describe.OrthographicWrapper():
            >>> with orth:
             ..     plot_3d_tensor_as_3d_plot(ar) 
            >>> plot_3d_tensor_as_3d_plot(ar)

    """
    def __init__(self):
        pass
    def __enter__(self):
        from mpl_toolkits.mplot3d import proj3d
        def orthogonal_proj(zfront, zback):
            a = (zfront+zback)/(zfront-zback)
            b = -2*(zfront*zback)/(zfront-zback)
            # -0.0001 added for numerical stability as suggested in:
            # http://stackoverflow.com/questions/23840756
            return np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,a,b],
                                [0,0,-0.0001,zback]])
        if not hasattr(proj3d,'old_persp_transformation'):
            proj3d.old_persp_transformation = proj3d.persp_transformation
        proj3d.persp_transformation = orthogonal_proj
    def __exit__(self, eType, eValue, eTrace):
        from mpl_toolkits.mplot3d import proj3d
        plt.show() # we have to show all the plots before restoring the transformation
        if hasattr(proj3d,'old_persp_transformation'):
            proj3d.persp_transformation = proj3d.old_persp_transformation

def plot_3d_tensor_as_3d_plot(ar,ax=None,scale_ar=None,num_levels = 20, contour_cmap='coolwarm_r', contourf_cmap='gray', view=(25, 65)):
    """
        Until I come up with a 3d contour plot that shows the contours of a volume, 
        this function visualizes a sequence of images as contour plots stacked on top of each other.
        The sides of the plot show a projection onto the xz and yz planes (at index 0).

            ar:
                the array to visualize

            ax: (if available)
                the matplotlib axis (in projection='3d' mode) from eg. calling subplot
                if none is provided, the current axis will be converted to 3d projection

            scale_ar:
                the array that is used for scaling (usefull if comparing arrays or visualizing only a small section)

            num_levels: 
                number of levels of contours

            contour_cmap='coolwarm_r': 
                color map used for line contours

            contourf_cmap='gray': 
                color map used for surface contour

            view:
                tuple of two floats that give the azimuth and angle of the projection


        returns:

            axis that was used for plotting


    """
    from mpl_toolkits.mplot3d import proj3d
    if ax is None:
        ax = plt.gca(projection='3d')
    if scale_ar is None:
        scale_ar = ar
    k_range = 0.5*np.max(scale_ar)-np.min(scale_ar)
    X, Y = np.meshgrid(np.arange(ar.shape[1]), np.arange(ar.shape[2]))
    levels = np.linspace(np.min(scale_ar), np.max(scale_ar), num_levels)
    for i,k in enumerate(ar):
        alpha = 100.0*(np.std(k)+0.003)
        if alpha > 1.0:
            alpha = 1.0
        if alpha < 0.0:
            alpha = 0.0
        k_scale = 1.0
        ax.contourf(X, Y, i+k_scale*k, cmap=contourf_cmap, 
                   levels=i+k_scale*levels,alpha=0.5*alpha)
        ax.contour(X, Y,i-k_scale*k/k_range, cmap=contour_cmap, 
                   levels=i-k_scale*levels/k_range,alpha=1.0,linewidths=3)
        side_line_color = [plt.cm.get_cmap(contour_cmap)(1.0-(np.mean(k)-np.min(scale_ar))/k_range)]
        cset = ax.contour(X, Y, i*np.ones_like(k), zdir='x', offset=0, cmap=None, colors='k',alpha=0.25)
        cset = ax.contour(X, Y, i*np.ones_like(k), zdir='y', offset=0, cmap=None, colors='k',alpha=0.25)
        cset = ax.contour(X, Y, i-k_scale*k/k_range, zdir='x', offset=0, cmap=None, colors=side_line_color,alpha=0.5*alpha)
        cset = ax.contour(X, Y, i-k_scale*k/k_range, zdir='y', offset=0, cmap=None, colors=side_line_color,alpha=0.5*alpha)
    ax.view_init(*view)
    ax.set_zlabel('time')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zticks(range(ar.shape[0]),minor=False)
    ax.invert_zaxis()
    return ax

def describe_layer_with_html(layer, max_depth = 3,wrap_in_html=False):
    from IPython.display import HTML
    s = ''
    def make_table(title,lst, line_style):
        from IPython.display import HTML
        lst = list(lst)
        sub_string = ''
        if len(lst) > 0:
            sub_string += '<div style="font-weight: bold;">'+title+'</div>'
            for name, val in lst:
                sub_string += '<div style="'+line_style+'">'+str(name)+' <i>('+str(val.__class__.__name__)
                if hasattr(val,'size'):
                    if len(val.size()) == 1 and val.size()[0] == 1:
                        sub_string += ' <i style="font-size: 80%;">scalar value</i>'
                    else:
                        sub_string += ' <i style="font-size: 80%;">'+str(' x '.join([str(int(i)) for i in val.size()]))+'</i>'
                if hasattr(val,'get'):
                        v = val.get()
                        if hasattr(v,'size'):
                            sub_string += ' <i style="font-size: 80%;">'+str(' x '.join([str(int(i)) for i in v.size()]))+'</i>'
                        else:
                            sub_string += ' <i style="font-size: 80%;">'+str(v)+'</i>'
                sub_string += ')</i>'
                if hasattr(val,'doc'):
                    sub_string += '<div style="font-size: 80%;">'+str(val.doc)+'</div>'
                sub_string += '</div>'
        return sub_string

    def r(model,depth):
        sub_string =''
        sub_string +='<div class="doc_string">'
        if hasattr(model,'__doc__'):
            try:
                sub_string += '<br />'.join(model.__doc__.strip().split('\n')[:1])+''
            except:
                pass
        sub_string +=  '&nbsp;&nbsp;<tiny style="font-size:80%;">(see full doc string with `help()`'
        if hasattr(model,'__doc_link__'):
            sub_string +=' or browse the documentation for <a href="'+model.__doc_link__+'">'+str(model.__class__.__name__)+'</a>'
        else:
            cls = str(model.__class__.__module__)+'.'+str(model.__class__.__name__)
            if cls.startswith('convis'):
                sphinx_path = "https://jahuth.github.io/convis/search.html?q="
                for doc_package,doc_url in doc_urls:
                    if cls.startswith(doc_package):
                        sphinx_path = doc_url
                        break
                sub_string +=' or browse the documentation for <a href="'+sphinx_path+cls+'" style="color:#222;">'+str(model.__class__.__name__)+'</a>'
        sub_string +=  ')</tiny>'
        sub_string += '</div>'
        if hasattr(model,'_state') and len(model._state) > 0:
            sub_string += make_table('State', [(s,getattr(model,s,None)) for s in model._state], 'margin:1px; padding:2px; background-color: #e3eb91; color: #444; margin-left:10px;')
        if hasattr(model,'_named_variables') and len(model._named_variables) > 0:
            sub_string += make_table('Variables', model._named_variables.items(), 'margin:1px; padding:2px; background-color: #75bea3; color: #444; margin-left:10px;')
        if len(list(model.named_children())) > 0:
            sub_string += '<div style="font-weight: bold;">Modules</div>'
            for mod_name,mod in list(model.named_children()):
                if mod_name is '':
                    continue
                bg = '#eae2e5' if depth%2==0 else '#7a7275'
                sub_string += '<div style="margin:1px; padding:2px; background-color: '+bg+'; color: #444; margin-left:10px;"><div>'
                sub_string += ' <b style:font-size:120%>'+str(mod_name)+'</b> ('+str(mod.__class__.__name__)+')</div>'
                if depth > 0:
                    #if isinstance(mod, Layer) or isinstance(mod, torch.nn.Module):
                    if hasattr(mod, 'named_children'):
                        sub_string += '<div>'+r(mod, depth-1)+'</div>'
                sub_string += '</div>'
        return sub_string
    s += r(layer,max_depth)
    if wrap_in_html:
        return HTML(s)
    return s