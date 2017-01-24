## Describing Variables 
#
# as dict, str or HTML
#
### Helper functions to deal with annotated variables

import numpy as np

plotting_possible = False
plotting_exceptions = []
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

def describe(v,**kwargs):
    return _Descriptor(v,**kwargs)

class _Descriptor(object):
    def __init__(self,v,**kwargs):
        self.v = v
        self.kwargs = kwargs
    def __repr__(self):
        return repr(describe_dict(self.v))
    def __str__(self):
        return str(describe_dict(self.v))
    def _repr_html_(self):
        return describe_html(self.v,wrap_in_html=False,**self.kwargs)

def describe_dict(v):
    if type(v) in [list, tuple] or hasattr(v,'__iter__'):
        try:
            return [describe_text(vv) for vv in v]
        except:
            # Tensor Variables love to raise TypeErrors when iterated over
            pass
    d = {}
    for k in ['name','simple_name','doc','config_key','optimizable','node','save','init','get','set']:
        if hasattr(v,k):
            d[k] = getattr(v,k)
    try:
        d['value'] = v.get_value()
    except:
        pass
    try:
        d['got'] = v.get(tu.create_context_O(v))
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
                    if preamble is False:
                        return img
                    return "Numpy array "+str(t.shape)+"<br/>"+img+legend            
            elif len(t.shape) == 5:
                # we assume its actually 3d with extra dimensions
                if t.shape[0] == 1 and t.shape[2] == 1:
                    return "Numpy array "+str(t.shape)+"<br/>"+_tensor_to_html(t[0,:,0,:,:],preamble=False,**kwargs)
                else:
                    return '5D tensor with too large first or third dimensions!'
        return 'Numpy Array ('+str(t.shape)+')'
    else:
        if preamble is False:
            return str(t[0])
        return str(type(t))+': '+str(t)

on_click_toggle =  """onclick='$(this).parent().children(".description_content").toggle();$(this).parent().children(".description_content_replacer").toggle();'"""

def save_name(n):
    return n.replace(' ', '_').replace('-', '_').replace('+', '_').replace('*', '_').replace('&', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '')
def full_path(v):
    return '_'.join([save_name(p.name) for p in v.path])

def describe_html(v,wrap_in_html=True,**kwargs):
    from IPython.display import HTML
    import html, uuid
    ##       
    # Handeling other datatypes
    #
    if type(v) == int or type(v) == float:
        s = str(v)
        if not wrap_in_html:
            return s
        return HTML(s)
    elif type(v) == np.ndarray:
        s = _tensor_to_html(v,**kwargs)
        if not wrap_in_html:
            return s
        return HTML(s)
    if type(v) in [dict] or hasattr(v,'__iteritems__'):
        uid = uuid.uuid4().hex
        s = "<div class='convis_description list'>"
        iteration = v.__iteritems__() if hasattr(v,'__iteritems__') else v.iteritems()
        s += "<b id="+uid+" "+on_click_toggle+" >+</b> &nbsp; &nbsp; "
        for (k,vv) in v.__iteritems__():
            s += '<a href="#'+uid+save_name(k)+'">'+k+'</a>  '
        s += "<div class='description_content_replacer' style='border-left: 4px solid #f0f0f0; padding-left: 5px; display: none;'>(&#8230;)</div>"
        s += "<div class='description_content' style='border-left: 4px solid #f0f0f0; padding-left: 5px;'>"
        iteration = v.__iteritems__() if hasattr(v,'__iteritems__') else v.iteritems()
        for (k,vv) in v.__iteritems__():
            s += "<div class='convis_description dict_item'><b id="+uid+save_name(k)+" "+on_click_toggle+" >"+str(k)+"</b><a href='#"+uid+"''>&#8617;</a>"
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
            s += "<div class='description_content_replacer' style='border-left: 4px solid #f0f0f0; padding-left: 5px; display: none;'>(&#8230;)</div>"
            s += "<div class='description_content' style='border-left: 4px solid #f0f0f0; padding-left: 5px;'>"
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
    for k in ['name','simple_name','doc','config_key','optimizable','node','save','init','get','set','variable_type']:
        if hasattr(v,k):
            d[k] = getattr(v,k)
    name = str(d.get('name','')) # optional: None handling
    simple_name = str(d.get('simple_name',''))
    s = """<div class='convis_description variable'><b """+on_click_toggle+""">"""+d.get('variable_type','')+""": """+simple_name+""" ("""+name+""")</b>"""
    # default: show everything, hide on click;
    s += "<div class='description_content_replacer' style='border-left: 2px solid #eee; padding-left: 5px; display: none;'>(&#8230;)</div>"
    s += "<div class='description_content' style='border-left: 2px solid #eee; padding-left: 5px;'>"
    if hasattr(v,'path'):
        s += "<small>" + full_path(v) + "</small>"
    if hasattr(v,'doc') and getattr(v,'doc') != '':
        s += '<p class="doc" style="padding:2px;">'+getattr(v,'doc')+'</p>'
    for k in ['config_key','optimizable','node','save','init','get','set']:
        if hasattr(v,k):
            s+= '<div><b>'+str(k)+'</b>: <tt>'+html.escape(str(getattr(v,k)))+'</tt></div>'
    try:
        if hasattr(v,'get_value'):
            s+= '<b>value</b>: ' + str(_tensor_to_html(v.get_value(),title=name,**kwargs))
    except Exception as e:
        s+= '<b>value</b>: ' + str(e)
        pass
    try:
        s+= '<b>got</b>: ' + _tensor_to_html(v.get(tu.create_context_O(v)),title=name,**kwargs)
    except:
        pass
    s += """</div>"""
    s += """</div>"""
    if not wrap_in_html:
        return s
    return HTML(s)