try:
    import new
except:
    new = False
from .misc_utils import unique_list
from .o import O, Ox, save_name
import numpy as np
import copy

replaceable_theano_vars = []#[TensorVariable,ScalarSharedVariable]

global_lookup_table = {}
only_use_lookup_table = True

if '__convis_global_lookup_table' in globals():
    global_lookup_table = globals()['__convis_global_lookup_table']
else:
    globals()['__convis_global_lookup_table'] = global_lookup_table

def get_convis_attribute(o,k,default):
    try:
        return getattr(o,k,default)
    except:
        return default
def has_convis_attribute(o,k):
    try:
        return hasattr(o,k)
    except:
        return False
def set_convis_attribute(o,k,v):
    setattr(o,k,v)

def full_path(v):
    return '_'.join([save_name(p) for p in get_convis_attribute(v,'path',[v])])

class ResolutionInfo(object):
    def __init__(self,pixel_per_degree=10.0,steps_per_second=1000.0,input_luminosity_range=1.0,filter_epsilon = 0.0001):
        """
            A resolution object tells a model how the dimensions of a
            discreet filter relate to the input in terms of:

                `pixel_per_degree`: the number of pixel (x or y) that correspond to one visual degree
                `steps_per_second`: the number of time steps that correspond to one second

            The functions `steps_to_seconds` and `seconds_to_steps`
            convert a float value from timesteps to seconds or vice versa.
            The functions `pixel_to_degree` and `degree_to_pixel`
            convert a float value from pixel to visual degree or vice versa.

            The attributes `var_pixel_per_degree` and `var_steps_per_second`
            provide theano variables that are updated every time either
            numerical attribute is changed.

            Use as such::

                res = convis.variables.Resolution()
                value_in_pixel = convis.as_parameter(100.0,'value_in_pixel')
                value_in_degree = convis.as_parameter(10.0,'value_in_degree')
                new_value_in_degree = value_in_pixel / res.var_pixel_per_degree
                new_value_in_pixel = value_in_degree * res.var_pixel_per_degree


            Please note: The singular of pixel is used for naming these functions:
            "pixel" and not "pixels". But to be compatible with
            VirtualRetina, there is one exception: The configuration value
            in VirtualRetina Configuration objects is named `pixels-per-degree`.

        """
        self._pixel_per_degree = pixel_per_degree
        self._steps_per_second = steps_per_second
        self.input_luminosity_range = input_luminosity_range
        self.filter_epsilon = filter_epsilon
    @property
    def pixel_per_degree(self):
        if self._pixel_per_degree is None:
            return default_resolution.pixel_per_degree
        return self._pixel_per_degree
    @pixel_per_degree.setter
    def pixel_per_degree(self,v):
        v = float(v)
        self._pixel_per_degree = v
    @property
    def steps_per_second(self):
        if self._steps_per_second is None:
            return default_resolution._steps_per_second
        return self._steps_per_second
    @steps_per_second.setter
    def steps_per_second(self,v):
        v = float(v)
        self._steps_per_second = v
    def degree_to_pixel(self,degree):
        if self.pixel_per_degree is None:
            return default_resolution.degree_to_pixel(degree)
        return float(degree) * self.pixel_per_degree
    def pixel_to_degree(self,pixel):
        if self.pixel_per_degree is None:
            return default_resolution.pixel_to_degree(pixel)
        return float(pixel) / self.pixel_per_degree
    def seconds_to_steps(self,t):
        if self.steps_per_second is None:
            return default_resolution.seconds_to_steps(t)
        return float(t) * self.steps_per_second
    def steps_to_seconds(self,steps):
        if self.steps_per_second is None:
            return default_resolution.steps_to_seconds(steps)
        return float(steps) / self.steps_per_second

default_resolution = ResolutionInfo(10.0,1000.0,1.0,filter_epsilon=0.001)

import torch

class Variable(torch.autograd.Variable):
    _is_convis_variable = True
    def __new__(self,x, **kwargs):
        if type(x) in [int, float]:
            x = np.array([x])
        for k,v in kwargs.items():
            setattr(self,k,v)
        return super(Variable, self).__new__(self,torch.Tensor(x))
    def __init__(self,x, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

class State(Variable):
    def __new__(self,x, **kwargs):
        return super(State, self).__new__(self,x)
    def __init__(self,x, **kwargs):
        super(State, self).__init__(x,**kwargs)

class Parameter(Variable,torch.nn.Parameter):
    def __new__(self,x, **kwargs):
        if type(x) in [int, float]:
            x = np.array([x])
        return super(Parameter, self).__new__(self,torch.Tensor(x))
    def __init__(self,x,default=None, **kwargs):
        #if type(x) in [int,float]:
        #    x = np.array([x])
        #super(Parameter, self).__init__(torch.Tensor(x))
        if default is not None:
            self.default = default
        else:
            self.default = np.copy(x)
        super(Parameter, self).__init__(x,**kwargs)
    @property
    def shape(self):
        return self.data.shape
    def set(self,v):
        if type(v) is None:
            return
        if type(v) is str:
            if '.' in v:
                v = float(v)
            else:
                v = int(v)
        if type(v) in [int, float]:
            v = np.array([v])
        self.data = torch.Tensor(v)

def as_parameter(x,**kwargs):
    return Parameter(x, **kwargs)

def is_variable(x):
    try:
        return hasattr(x,'_is_convis_variable')
    except:
        return False

def dict_with_dots_to_hierarchical_dict(d):
    d = copy.copy(d)
    # move the empty string key (the module itself)
    # to another key
    if '' in d.keys():
        d['_self'] = d['']
        del d['']
    for k in d.keys():
        if '.' in k:
            val = d[k]
            del d[k]
            ks = k.split('.')
            if ks[0] in d:
                if type(d[ks[0]]) == dict:
                    d[ks[0]].update(dict_with_dots_to_hierarchical_dict({'.'.join(ks[1:]): val}))
                else:
                    old_val = d[ks[0]]
                    d[ks[0]] = {'_self': d[ks[0]]}
                    d[ks[0]].update(dict_with_dots_to_hierarchical_dict({'.'.join(ks[1:]): val}))
                    #raise Exception('Key collision! %s and %s'%(ks[0],k))
            else:
                d[ks[0]] = dict_with_dots_to_hierarchical_dict({'.'.join(ks[1:]): val})
    return d

def create_Ox_from_torch_iterator_dicts(iterator,doc=''):
    """
        Takes a dictionary (or iterator with key,value pairs)
        with dots in the keys and returns a hierarchical object.

            >>> p = convis.variables.create_Ox_from_torch_iterator_dicts(retina.named_parameters())
                # the same as:
            >>> p = Ox(**dict_with_dots_to_hierarchical_dict(dict(retina.named_parameters())))
            >>> p.<tab complete>
            >>> p.bipolar.<tab complete>
            >>> p.bipolar.g_leak


        :class:`convis.o.Ox` objects have a few special attributes (with leading underscores)
        that help finding nodes in large hierarchies

            >>> p._all # lists everything with underscore 
                       # separating hierarchies (ie. similar
                       # to `.named_parameters()`, but each works
                       # as an attribute instead of a string
                       # key)
            >>> p._search.g_leak.<tab complete>
            >>> p._search.g_leak.bipolar_g_leak

        See also
        --------

        convis.o.Ox
    """
    return Ox(_doc=doc,**dict_with_dots_to_hierarchical_dict(dict(iterator)))

def create_hierarchical_dict(vs,pi=0,name_sanitizer=save_name):
    """
        pi: "path i" offset in the path

            The path will only be used from element pi onwards
    """
    o = {}
    paths = unique_list([get_convis_attribute(v,'path',[v])[pi] for v in vs if len(get_convis_attribute(v,'path',[v])) > pi+1])
    leaves = unique_list([v for v in vs if len(get_convis_attribute(v,'path',[v])) == pi+1])
    for p in paths:
        o.update(**{name_sanitizer(get_convis_attribute(p,'name')): create_hierarchical_dict([v for v in vs if has_convis_attribute(v,'path',[v]) 
                                                                and len(get_convis_attribute(v,'path',[v])) > pi 
                                                                and get_convis_attribute(v,'path',[v])[pi] == p], pi+1)})
    for p in paths:
        o[name_sanitizer(get_convis_attribute(p,'name'))]['_original'] = p 
    for l in leaves:
        o.update(**{name_sanitizer(get_convis_attribute(l,'name')): l})
    return o

def create_hierarchical_Ox(vs,pi=0,name_sanitizer=save_name):
    return Ox(**create_hierarchical_dict(vs,pi,name_sanitizer=name_sanitizer))

def create_hierarchical_dict_with_nodes(vs,pi=0,name_sanitizer=save_name):
    """
        name_sanitizer: eg. convis.base.save_name or str
    """
    o = {}
    paths = unique_list([get_convis_attribute(v,'path',[v])[pi] for v in vs if len(get_convis_attribute(v,'path',[v])) > pi+1])
    leaves = unique_list([v for v in vs if len(get_convis_attribute(v,'path',[v])) == pi+1])
    for p in paths:
        o.update(**{p: create_hierarchical_dict_with_nodes([v for v in vs if 
                                                        has_convis_attribute(v,'path',[v]) 
                                                        and len(get_convis_attribute(v,'path',[v])) > pi 
                                                        and get_convis_attribute(v,'path',[v])[pi] == p], pi+1)})
    for l in leaves:
        o.update(**{name_sanitizer(get_convis_attribute(l,'name')): l})
    return o


def unindent(text):
    from textwrap import dedent
    if text == '':
        return text
    lines = text.split('\n')
    if lines[0] == lines[0].rstrip():
        return lines[0]+'\n'+dedent('\n'.join(lines[1:]))
    else:
        return dedent(text)

def add_kwargs_to_v(v,**kwargs):
    update_convis_attributes(v, kwargs)
    if 'doc' in kwargs.keys():
        set_convis_attribute(v,'doc', unindent(kwargs.get('doc','')))
    return v

def raise_exception(e):
    raise e
def create_context_O(var=None, **kwargs):
    """
        This function creates the 'typical' context that annotated variables can expect when defining `init` functions.

            * `var`: the variable itself
            * `node`: the node that wraps this part of the graph (providing eg. configuration options)
            * `get_config`: a function of `node` that provides the configuration dictionary local to this node
            * `model`: the model providing global options such as `pixel_to_degree()` and `seconds_to_step()`
    
        During execution:

            * `input`: the variable that is fed as input to the model (not the node!)

        Further, if the variable has a `config_key` and a `config_default` field,
        two short cut functions retrieve and save the configuration value to minimize code redundancy:

            * `value_from_config()`: retrieves the configuration value from `node`, falling back to `config_default` if no option is provided
            * `value_to_config(v)`: calls the `set_config` method of the `node` to save a configuration value (eg. when updated when optimizing)

        To use these values in an `init` function eg::

            as_parameter(T.iscalar("k"),init=lambda x: x.input.shape[0])
    """
    if var is None:
        return O(resolution=default_resolution)(**kwargs)
    node = get_convis_attribute(var,'node')
    model = None
    get_config = None
    get_config_value = None
    if node is not None:
        model = node.get_model()
        get_config = node.get_config
        get_config_value = node.get_config_value
    config_key = get_convis_attribute(var,'config_key','')
    if has_convis_attribute(var, 'config_key') and hasattr(var,'get_value'):
        return O(var=var,node=node,model=model,resolution=getattr(node,'resolution',default_resolution),
                 get_config=get_config,
                 get_config_value=get_config_value,
                 value_from_config=lambda: node.get_config_value(config_key,get_convis_attribute(var,'config_default',var.get_value())),
                 value_to_config=lambda v: node.set_config_value(config_key,v))(**kwargs)
    elif has_convis_attribute(var, 'config_key') and has_convis_attribute(var, 'config_default'):
        return O(var=var,node=node,model=model,resolution=getattr(node,'resolution',default_resolution),
                 get_config=get_config,
                 get_config_value=get_config_value,
                 value_from_config=lambda: node.get_config_value(config_key,get_convis_attribute(var,'config_default',get_convis_attribute(var, 'config_default'))),
                 value_to_config=lambda v: node.set_config_value(config_key,v))(**kwargs)
    return O(var=var,node=node,model=model,get_config_value=get_config_value,get_config=get_config,resolution=getattr(node,'resolution',default_resolution),
             value_from_config=lambda: raise_exception(Exception('No config key and default value available. '+str(get_convis_attribute(var,'name'))+'\n')))(**kwargs)


def update(var,**kwargs):
    return get_convis_attribute(var,'update')(create_context_O(var,**kwargs))




"""
    Combining Virtual Parameters


"""

def is_callback(v):
    return type(v) in [CallbackParameter,VirtualParameter]
def get_if_callback(v):
    if is_callback(v):
        return v.get()
    return v

class IndirectParameter(object):
    _is_convis_variable = True
    def __init__(self,func,name,var=None,*dependencies,**kwargs_dependencies):
        self.func = func
        self.name = name
        self.var = var
        self.dependencies = dependencies
        for d in self.dependencies:
            if is_callback(d):
                d.call = self
        self.kwargs_dependencies = kwargs_dependencies
        for d in self.kwargs_dependencies.values():
            if is_callback(d):
                d.call = self
    def set(self,v=None):
        self.value = self.func(*[get_if_callback(d) for d in self.dependencies],
                               **dict([(k,get_if_callback(v)) for (k,v) in self.kwargs_dependencies.items()]))
        if self.var is not None:
            self.var.set(self.value)

class CallbackParameter(object):
    _is_convis_variable = True
    def __init__(self,func,name,call=None,value=None):
        self.value = value
        self.func = func
        self.call = call
        self.name = name
    def set(self,v):
        self.value = self.func(v)
        if self.call is not None:
            self.call.set()
    def get(self):
        return self.value

class FakeTensor(object):
    def __init__(self):
        pass
    def cuda(self,x=None):
        return self
    def cpu(self,x=None):
        return self

class VirtualParameter(object):
    """
        VirtualParameters can generate parameter values
        from a dependency structure of other parameters.
        
        Example::
        
            a = VirtualParameter(float,value=0.1)
            b = VirtualParameter(int,value=0)
            v = VirtualParameter(convis2.numerical_filters.exponential_filter_5d).set_callback_arguments(tau=a,n=b)
            a.set(0.01) # updating this parameter causes the function of v to be reevaluated 
            plot(v.get()[0,:,0,0,0])
            b.set(2) # the function is reevaluated again
            plot(v.get()[0,:,0,0,0])
        
    """
    _is_convis_variable = True
    data = FakeTensor()
    def __init__(self,func=None,var=None,call=None,value=None,dependencies=[],kwargs_dependencies={},**kwargs):
        self.func = func
        self._grad = None
        self.name = ''
        self.var = var
        self.callbacks = []
        self.value = value
        self.dependencies = dependencies
        for d in self.dependencies:
            if is_callback(d):
                d.callbacks.append(self)
        self.kwargs_dependencies = kwargs_dependencies
        for d in self.kwargs_dependencies.values():
            if is_callback(d):
                d.callbacks.append(self)
        for k,v in kwargs.items():
            setattr(self,k,v)
    def __repr__(self):
        return 'VirtualParameter('+str(self.func)+(
            ' with variable '+str(self.var) if self.var is not None else '')+')'
    def set_callback_arguments(self,*dependencies,**kwargs_dependencies):
        self.dependencies = dependencies
        for d in self.dependencies:
            if is_callback(d):
                d.callbacks.append(self)
        self.kwargs_dependencies = kwargs_dependencies
        for d in self.kwargs_dependencies.values():
            if is_callback(d):
                d.callbacks.append(self)
        return self
    def set_variable(v):
        self.var = v
    def set(self,v):
        if self.func is not None:
            self.value = self.func(v)
        else:
            self.value = v
        for c in self.callbacks:
            if hasattr(c,'update'):
                c.update()
            else:
                c()
        if self.var is not None:
            if hasattr(self.var,'set'):
                self.var.set(self.value)
            elif hasattr(self.var,'data'):
                if not hasattr(self.value,'shape'):
                    self.var.data[0] = self.value
                elif len(self.var.data.shape) == 1:
                    self.var.data[:] = self.value
                elif len(self.var.data.shape) == 2:
                    self.var.data[:,:] = self.value
                elif len(self.var.data.shape) == 3:
                    self.var.data[:,:,:] = self.value
                elif len(self.var.data.shape) == 4:
                    self.var.data[:,:,:,:] = self.value
                elif len(self.var.data.shape) == 5:
                    self.var.data[:,:,:,:,:] = self.value
                else:
                    raise Exception('Do not know how to set variable '+str(self.var)+' with value '+str(self.value))
    def get(self):
        return self.value
    def update(self):
        self.value = self.func(*[get_if_callback(d) for d in self.dependencies],
                               **dict([(k,get_if_callback(v)) for (k,v) in self.kwargs_dependencies.items()]))
        if self.var is not None:
            self.var.set(self.value)