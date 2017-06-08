import theano
import new
from debug import *
from misc_utils import unique_list
from o import O, Ox, save_name
from theano.tensor.var import TensorVariable
from theano.tensor.sharedvar import ScalarSharedVariable
import numpy as np
replaceable_theano_vars = []#[TensorVariable,ScalarSharedVariable]

global_lookup_table = {}
only_use_lookup_table = True

if '__convis_global_lookup_table' in globals():
    global_lookup_table = globals()['__convis_global_lookup_table']
else:
    globals()['__convis_global_lookup_table'] = global_lookup_table

def full_path(v):
    return '_'.join([save_name(p) for p in get_convis_attribute(v,'path',[v])])

def is_convis_var(v):
    if hasattr(v,'name') and type(v.name) is str:
        if '!' in v.name:
            return True
    return hasattr(v,'_convis_lookup')

def has_convis_attribute(v,key,default=None):
    if not is_convis_var(v):
        return hasattr(v,key)
    global global_lookup_table
    if key in get_convis_attribute_dict(v):
        return True
    return hasattr(v,key)
    
def get_convis_attribute(v,key,default=None):
    if not is_convis_var(v):
        return getattr(v,key,default)
    if key is 'name':
        name = str(v.name).split('!')[-1]
        if len(name) == 0:
            return None
        return name
    d = get_convis_attribute_dict(v)
    return d.get(key, getattr(v, key, default))

def get_convis_key(v):
    if hasattr(v,'_convis_lookup'):
        if not v._convis_lookup in global_lookup_table:
            global_lookup_table[v._convis_lookup] = {}
        return v._convis_lookup
    if hasattr(v,'name') and v.name is not None and '!' in v.name:
        s = v.name.split('!')[0]
        if not s in global_lookup_table:
            global_lookup_table[s] = {}
        return s
    return create_convis_key(v)

def create_convis_key(v):
    global global_lookup_table
    new_key = len(global_lookup_table.keys())
    while str(new_key) in global_lookup_table.keys():
        new_key += 1
    if hasattr(v,'name'):
        if v.name is None:
            v.name = str(new_key)+'!'
        else:
            v.name = str(new_key)+'!'+str(v.name)
    try:
        v.__is_convis_var = True
        v._convis_lookup = str(new_key)
    except:
        pass
    global_lookup_table[str(new_key)] = {'_id': str(new_key)}
    return str(new_key)

def get_convis_attribute_dict(v):
    try:
        return global_lookup_table[get_convis_key(v)]
    except:
        return None

def set_convis_attribute(v,key,value):
    if v is None:
        return
    global global_lookup_table
    global only_use_lookup_table
    global convis_attributes
    d = get_convis_attribute_dict(v)
    if only_use_lookup_table and key in convis_attributes and key not in ['__is_convis_var', 'name']:
        d[key] = value
        return
    if key is 'name':
        if '_id' in d:
            v.name = str(d['_id'])+'!'+value
        else:
            v.name = value
        return
    setattr(v,key,value)

def update_convis_attributes(v,new_d):
    #if v is None:
    #    return
    #global global_lookup_table
    #global only_use_lookup_table
    if only_use_lookup_table:
        d = get_convis_attribute_dict(v)
        d.update(new_d)
    else:
        for (key,value) in d.items():
            setattr(v,key,value)

def make_variable(ret):
    """
        Attempts to wrap `ret` into a proxy class, such that convis variables as
        well as thenao attributes are available
    """
    if type(ret) in replaceable_theano_vars:
        return Variable(ret)
    return ret

class ConfigParameter(object):
    def __init__(self,name,default,type=None):
        self.name = name
        self.config = None
        self.var = None
        self.default = default
        if type is not None:
            self.type = type
        self.type = lambda x: x
    def set(self, value):
        self.config[self.name] = self.type(value)
        self.var.set_value(self.type(value))
    def var_to_config(self, value):
        self.config[self.name] = self.type(self.var.get_value())
    def config_to_var(self, value):
        if config is not None:
            self.var.set_value(self.type(self.config.get(self.name,default)))
        else:
            self.var.set_value(self.type(default))


class Variable(object):
    __slots__ = ["_var","_info", "__weakref__","_convis_lookup"]
    def __init__(self, obj):
        """
            A `convis.Variable` object wraps a theano variable
            and provides direct access to its convis attributes
            which are otherwise hidden.

            Note: this object is *sometimes* interchangeable with
            theano variables::

                v = Variable(as_parameter(0.0))
                v.name = "v"
                v.doc = "A variable named `v`"
                v_squared = v**2 # returns a convis variable.
                v_sum = T.sum(v) # returns a theano variable!

            The value returned from theano functions will not be a 
            convis Variable.

            To be absolutely certain that a operation
            uses the associated theano variable use the `._var`
            attribute

            At the moment it is not advised to use this object inplace
            of theano variables, since that can easily lead to confusion.
            

        """
        object.__setattr__(self, "_var", obj)
        #print obj, str(obj.owner)
        object.__setattr__(self, "_convis_lookup", get_convis_key(obj))
        object.__setattr__(self, "_info", get_convis_attribute_dict(obj))
    def _as_TensorVariable(self):
        return object.__getattribute__(self, "_var")
    def __getattribute__(self, name):
        if name in ['_var','_info']:
            return object.__getattribute__(self, name)
        if name == 'name':
            return getattr(object.__getattribute__(self, "_var"), name)
        if name in convis_attributes:
            if name in object.__getattribute__(self, "_info").keys():
                return object.__getattribute__(self, "_info").get(name)
        return getattr(object.__getattribute__(self, "_var"), name)
    def __delattr__(self, name):
        if name in convis_attributes:
            if name in object.__getattribute__(self, "_info").keys():
                return object.__getattribute__(self, "_info").remove(name)
        delattr(object.__getattribute__(self, "_var"), name)
    def __setattr__(self, name, value):
        if name == 'name':
            set_convis_attribute(object.__getattribute__(self, "_var"), name, value)
        else:
            if name in convis_attributes:
                if name in object.__getattribute__(self, "_info").keys():
                    object.__getattribute__(self, "_info")[name] = value
            setattr(object.__getattribute__(self, "_var"), name, value)
    def __nonzero__(self):
        return bool(object.__getattribute__(self, "_var"))
    def __str__(self):
        return str(object.__getattribute__(self, "_var"))
    def __unicode__(self):
        return unicode(object.__getattribute__(self, "_var"))
    def __repr__(self):
        return repr(object.__getattribute__(self, "_var"))
    def __hash__(self):
        return hash(object.__getattribute__(self, "_var"))
    def __dir__(self):
        my_convis_attributes = filter(lambda x: has_convis_attribute(self, x), convis_attributes)
        return my_convis_attributes + dir(object.__getattribute__(self, "_var"))
    @classmethod
    def __instancecheck__(self, instance):
        try:
            return isinstance(instance, type(object.__getattribute__(instance, "_var")))
        except:
            return isinstance(instance, self)
    _special_names = [
        '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__', 
        '__contains__', '__delitem__', '__delslice__', '__div__', '__divmod__', 
        '__eq__', '__float__', '__floordiv__', '__ge__', '__getitem__', 
        '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__',
        '__idiv__', '__idivmod__', '__ifloordiv__', '__ilshift__', '__imod__', 
        '__imul__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', 
        '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', 
        '__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', 
        '__neg__', '__oct__', '__or__', '__pos__', '__pow__', '__radd__', 
        '__rand__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__', 
        '__repr__', '__reversed__', '__rfloorfiv__', '__rlshift__', '__rmod__', 
        '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', 
        '__rtruediv__', '__rxor__', '__setitem__', '__setslice__', '__sub__', 
        '__truediv__', '__xor__', 'next','__rfloorfiv__'
    ]
    
    @classmethod
    def _create_class_proxy(cls, theclass):
        """creates a proxy for the given class"""
        def make_method(name):
            def method(self, *args, **kw):
                args = [getattr(a,'_var',a) for a in args]
                kw = dict((k,getattr(v,'_var',v)) for k,v in kw.items())
                ret = getattr(object.__getattribute__(self, "_var"), name)(*args, **kw)
                if type(ret) in replaceable_theano_vars:
                    return Variable(ret)
                return ret
            return method
        namespace = {}
        for name in cls._special_names:
            if hasattr(theclass, name) and not hasattr(cls, name):
                namespace[name] = make_method(name)
            # else:
            #     pass
            #     # if name == '__eq__':
            #     #     def eq(self, other):
            #     #         return object.__getattribute__(self, "_var") == getattr(other,'_var',other) 
            #     #     namespace[name] = eq
            # if name == '__ne__':
            #     def ne(self, other):
            #         # this is never called?
            #         print 'testing for not equalness'
            #         return not object.__getattribute__(self, "_var") == getattr(other,'_var',other)  
            #     namespace[name] = ne
        return type("%s(%s)" % (cls.__name__, theclass.__name__), (cls,), namespace)
    
    def __new__(cls, obj, *args, **kwargs):
        """
        creates an proxy instance referencing `obj`. (obj, *args, **kwargs) are
        passed to this class' __init__, so deriving classes can define an 
        __init__ method of their own.
        note: _class_proxy_cache is unique per deriving class (each deriving
        class must hold its own cache)
        """
        try:
            cache = cls.__dict__["_class_proxy_cache"]
        except KeyError:
            cls._class_proxy_cache = cache = {}
        try:
            theclass = cache[obj.__class__]
        except KeyError:
            cache[obj.__class__] = theclass = cls._create_class_proxy(obj.__class__)
        ins = object.__new__(theclass)
        return ins

class ResolutionInfo(object):
    def __init__(self,pixel_per_degree=10.0,steps_per_second=1000.0,input_luminosity_range=1.0,filter_epsilon = 0.001):
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


            Please note: The plural of pixel used for naming these functions 
            is "pixel" and never "pixels". But to be compatible with
            VirtualRetina, there is one exception: The configuration value
            in VirtualRetina Configuration objects is named `pixels-per-degree`.

        """
        self._pixel_per_degree = pixel_per_degree
        self._steps_per_second = steps_per_second
        self.input_luminosity_range = input_luminosity_range
        self.filter_epsilon = filter_epsilon
        self.var_pixel_per_degree = theano.shared(float(self.pixel_per_degree))
        self.var_steps_per_second = theano.shared(float(self.steps_per_second))
        self.var_input_luminosity_range = theano.shared(float(self.input_luminosity_range))
        self.var_filter_epsilon = theano.shared(self.filter_epsilon)
    @property
    def pixel_per_degree(self):
        if self._pixel_per_degree is None:
            return default_resolution.pixel_per_degree
        return self._pixel_per_degree
    @pixel_per_degree.setter
    def pixel_per_degree(self,v):
        v = float(v)
        self._pixel_per_degree = v
        self.var_pixel_per_degree.set_value(v)
    @property
    def steps_per_second(self):
        if self._steps_per_second is None:
            return default_resolution._steps_per_second
        return self._steps_per_second
    @pixel_per_degree.setter
    def steps_per_second(self,v):
        v = float(v)
        self._steps_per_second = v
        self.var_steps_per_second.set_value(v)
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

default_resolution = ResolutionInfo(10.0,1000.0,1.0)


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

convis_attributes = ['name','node','path','__is_convis_var','variable_type','doc','root_of',
                    'state_out_state','state_init','state_in_state','_old_type',
                    'param_init','initialized','optimizable','config_key','config_default','save','get','_convis_lookup']


def get_convis_attributes(v):
    return O(**dict([(a,get_convis_attribute(v,a)) for a in convis_attributes if has_convis_attribute(v,a)]))

def get_convis_attributes_dict(v):
    return dict([(a,get_convis_attribute(v,a)) for a in convis_attributes if has_convis_attribute(v,a)])

#proxy objects are better?
#def v(v):
#    return O(**v.__dict__)(variable=v, **get_convis_attributes_dict(v))


def override_copy(v,actually_do_it=True):
    return v
    from copy import deepcopy
    import copy
    # Hopefully we won't need this.
    # If we inject functions into the variables, it is possible we get Pickle Errors on compile time!
    # def new_copy(self, name=None):
    #     """Return a symbolic copy and optionally assign a name.
    #     Does not copy the tags.
    #     Also copies convis specific attributes.
    #     """
    #     global convis_attributes
    #     copied_variable = theano.tensor.basic.tensor_copy(self)
    #     copied_variable.name = name
    #     if hasattr(self,'preserve_labels_on_copy'):
    #         for a in convis_attributes:
    #             if hasattr(self,a):
    #                 setattr(copied_variable,a,getattr(self,a))
    #         copied_variable.copied_from = self.v
    #     override_copy(copied_variable)
    #     return copied_variable
    def type_call(self,*args,**kwargs):
        new_v = self.__old_call__(*args,**kwargs)
        if hasattr(self,'v') and has_convis_attribute(self.v,'preserve_labels_on_copy'):
            for a in convis_attributes:
                if has_convis_attribute(self.v,a):
                    set_convis_attribute(new_v,a,getattr(self.v,a))
            set_convis_attribute(new_v,'copied_from',self.v)
        override_copy(new_v)
        return new_v
    def type_make_variable(self,name=None):
        new_v = self.Variable(self,name=name)
        if hasattr(self,'v') and has_convis_attribute(self.v,'preserve_labels_on_copy'):
            for a in convis_attributes:
                if has_convis_attribute(self.v,a):
                    set_convis_attribute(new_v,a,get_convis_attribute(self.v,a))
            set_convis_attribute(new_v,'copied_from',self.v)
        override_copy(new_v)
        return new_v
    #v.copy = new.instancemethod(new_copy, v, None)
    if not has_convis_attribute(v,'_old_type'):
        set_convis_attribute(v,'_old_type', v.type)
    v.type = copy.copy(v.type)
    if not hasattr(v.type,'__old_type__'):
        old_type = get_convis_attribute(v,'_old_type',v.type)
        v.type = copy.copy(v.type)
        v.type.__old_call__ = old_type.__call__
        v.type.__old_type__ = get_convis_attribute(v,'_old_type')
        #if hasattr(v.type,'v'):
        #    print v.type.v
        #    raise Exception("type already has a v!!")
        v.type.v = v
        v.type.__call__ = new.instancemethod(type_call, v.type, None)
        v.type.make_variable = new.instancemethod(type_make_variable, v.type, None)
    v.type.v = v
    if actually_do_it:
        v.preserve_labels_on_copy = True
    return v

# functions on theano variables
def as_state(v,out_state=None,init=None,name=None,**kwargs):
    get_convis_key(v)
    v.__is_convis_var = True
    set_convis_attribute(v,'variable_type','state')
    if out_state is not None:
        set_convis_attribute(v,'state_out_state',out_state)
    if init is not None:
        set_convis_attribute(v,'state_init',init)
    if name is not None:
        set_convis_attribute(v,'name',name)
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)
def as_out_state(v,in_state=None,init=None,name=None,**kwargs):
    get_convis_key(v)
    v.__is_convis_var = True
    set_convis_attribute(v,'variable_type','out_state')
    if in_state is not None:
        as_state(in_state,out_state=v,init=init)
        set_convis_attribute(v,'state_in_state',in_state)
    if name is not None:
        set_convis_attribute(v,'name',name)
    if get_convis_attribute(v,'name') is None:
        if in_state is not None and in_state.name is not None:
            set_convis_attribute(v,'name',in_state.name+'_out_state')
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)

def as_input(v,name=None,**kwargs):
    get_convis_key(v)
    v.__is_convis_var = True
    set_convis_attribute(v,'variable_type','input')
    if name is not None:
        set_convis_attribute(v,'name', name)
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)
def as_variable(v,name=None,**kwargs):
    get_convis_key(v)
    v.__is_convis_var = True
    if name is not None:
        set_convis_attribute(v,'name', name)
    if not hasattr(v,'variable_type'):
        # don't overwrite the variable type!
        set_convis_attribute(v,'variable_type','variable')
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)
def as_output(v,name=None,**kwargs):
    get_convis_key(v)
    v.__is_convis_var = True
    if not hasattr(v,'variable_type'):
        # is the output variable type even important?
        # in any case things can be eg. parameters and also outputs, so we don't want to overwrite this!
        set_convis_attribute(v,'variable_type','output')
    if name is not None:
        set_convis_attribute(v,'name', name)
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)

def as_parameter(v,init=None,name=None,**kwargs):
    cparm = None
    if type(v) in [int, float, np.ndarray, np.float, np.float16, np.float32, np.float64, np.float128,
    np.int, np.int0, np.int8, np.int16, np.int32, np.int64]:
        v = theano.shared(v)
    elif type(v) is ConfigParameter:
        cparm = v
        v = theano.shared(v.default)
        cparm.var = v
    get_convis_key(v)
    v.__is_convis_var = True
    set_convis_attribute(v,'variable_type','parameter')
    if cparm is not None:
        set_convis_attribute(v,'config_parameter',cparm)
    if init is not None:
        set_convis_attribute(v,'param_init',init)
    if name is not None:
        set_convis_attribute(v,'name', name)
    if hasattr(v,'set_value'):
        def update(self,x=O(),force=False):
            """
                When other variables depend on this variable to be up-to-date, they can call `update` to set it its value according to the current configuration.

                If a variable already had its 'initialized' attribute set, it will not be computed again.
                Also if the variable is flagged as 'optimizing', no changes will be made.

                The function needs to be supplied with a context that holds information about who the model is and what the input looks like (see `create_context_O`).

                Setting force=True will reinitialize the variable regardless of initialized and optimizing status.
            """
            if self in getattr(x,'update_trace',[]):
                # The variable was already updated. To avoid cycles we do not update again.
                warnings.warn("The variable was already updated. To avoid cycles we do not update again.", Warning)
                send_dbg('param.update',str(self.name)+' was attempted to be updated twice!',4)
                return self
            if not force and (get_convis_attribute(self, 'initialized', False) or has_convis_attribute(self,'optimizing')):
                # The variable was already updated. To avoid cycles we do not update again.
                send_dbg('param.update',str(self.name)+' was already initialized or is optimizing itself.',0)
                return self
            # this might entail recursion when param_init calls update of other variables:
            init = get_convis_attribute(self,'param_init',lambda x: x.old_value)
            if callable(init):
                new_val = init(create_context_O(self,old_value=self.get_value(),update_trace=getattr(x,'update_trace',[])+[self]))
            else:
                if has_convis_attribute(self,'config_key'):
                    new_val = x.value_from_config()
                else:
                    new_val = init
            #if self.name is not None and 'lambda' in self.name:
            #    print self.name, new_val
            send_dbg('param.update',str(self.name)+' updated itself from '+str(self.get_value())[:10]+' to '+str(new_val)[:10]+'.',2)
            self.set_value(new_val)
            set_convis_attribute(self,'initialized',True)
            return self
        set_convis_attribute(v,'update',new.instancemethod(update, v, None))
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)


def is_var(v):
    return is_convis_var(v)
def is_named_var(v):
    return not get_convis_attribute(v,'name') is None
def is_state(v):
    return has_convis_attribute(v,'variable_type') and get_convis_attribute(v,'variable_type') == 'state' and has_convis_attribute(v,'state_out_state')
def is_out_state(v):
    return has_convis_attribute(v,'variable_type') and get_convis_attribute(v,'variable_type') == 'out_state'
def is_input_parameter(v):
    return has_convis_attribute(v,'variable_type') and (not hasattr(v,'get_value')) and get_convis_attribute(v,'variable_type') == 'parameter'  and (not has_convis_attribute(v,'copied_from'))
def is_shared_parameter(v):
    return has_convis_attribute(v,'variable_type') and hasattr(v,'get_value') and get_convis_attribute(v,'variable_type') == 'parameter'
def is_parameter(v):
    return is_input_parameter(v) or is_shared_parameter(v)
def is_input(v):
    return has_convis_attribute(v,'variable_type') and get_convis_attribute(v,'variable_type') == 'input'
def are_states(vs):
    return filter(is_state,vs)
def are_out_states(vs):
    return filter(is_out_state,vs)
def are_parameters(vs):
    return filter(is_input_parameter,vs)
def are_inputs(vs):
    return filter(is_input,vs)
def is_scalar(v):
    """Returns True if v is an int, a float or a 0 dimensional numpy array or theano variable"""
    if type(v) in [int, float]:
        return True
    if type(v) is np.ndarray and hasattr(v,'shape') and len(v.shape) == 0:
        return True
    if hasattr(v,'type') and hasattr(v.type,'ndim') and v.type.ndim == 0:
        return True
    return False

def shared_parameter(fun,init_object=None,**kwargs):
    if callable(fun):
        if init_object is None:
            init_object = create_context_O()
        if not hasattr(init_object,'resolution'):
            init_object(resolution = default_resolution) # for initial filters we only want a single 1
        return as_parameter(theano.shared(fun(init_object)),
                            initialized = True,
                            init=fun,**kwargs)
    return as_parameter(theano.shared(fun),
                        initialized = True,
                        init=fun,**kwargs)


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
    if node is not None:
        model = node.get_model()
        get_config = node.get_config
        get_config_value = node.get_config_value
    config_key = get_convis_attribute(var,'config_key','')
    if has_convis_attribute(var, 'config_key') and hasattr(var,'get_value'):
        return O(var=var,node=node,model=model,resolution=getattr(model,'resolution',default_resolution),
                 get_config=get_config,
                 get_config_value=get_config_value,
                 value_from_config=lambda: node.get_config_value(config_key,get_convis_attribute(var,'config_default',var.get_value())),
                 value_to_config=lambda v: node.set_config_value(config_key,v))(**kwargs)
    elif has_convis_attribute(var, 'config_key') and has_convis_attribute(var, 'config_default'):
        return O(var=var,node=node,model=model,resolution=getattr(model,'resolution',default_resolution),
                 get_config=get_config,
                 get_config_value=get_config_value,
                 value_from_config=lambda: node.get_config_value(config_key,get_convis_attribute(var,'config_default',get_convis_attribute(var, 'config_default'))),
                 value_to_config=lambda v: node.set_config_value(config_key,v))(**kwargs)
    return O(var=var,node=node,model=model,get_config_value=get_config_value,get_config=get_config,resolution=getattr(model,'resolution',default_resolution),
             value_from_config=lambda: raise_exception(Exception('No config key and default value available. '+str(get_convis_attribute(var,'name'))+'\n')))(**kwargs)
