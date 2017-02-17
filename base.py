import litus
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pylab as plt
from theano.tensor.nnet.conv3d2d import conv3d
from theano.tensor.signal.conv import conv2d
import uuid
from . import retina_base
from . import theano_utils
from exceptions import NotImplementedError
from variable_describe import describe, describe_dict, describe_html, full_path, save_name
import new
import warnings
import datetime
from collections import OrderedDict

def f7(seq):
    """ This function is removing duplicates from a list while keeping the order """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]





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
    v.__dict__.update(kwargs)
    if 'doc' in kwargs.keys():
        v.__dict__['doc'] = unindent(kwargs.get('doc',''))
    return v

convis_attributes = ['name','node','path','__is_convis_var','variable_type','doc','root_of',
                    'state_out_state','state_init','state_in_state',
                    'param_init','initialized','optimizable','config_key','config_default','save','get']

def get_convis_attributes(v):
    return O(**dict([(a,getattr(v,a)) for a in convis_attributes if hasattr(v,a)]))

do_debug = False
do_replace_inputs = True

debug_messages = []

def send_dbg(channel,msg,log_level=0):
    global debug_messages
    debug_messages.append((channel,msg,log_level,datetime.datetime.now()))

def _filter_dbg_txt(search,txt):
    if search is None:
        return True
    if type(search) is list:
        for s in search:
            if not _filter_dbg_txt(s,txt):
                return False    
        return True
    return search in txt

def filter_dbg(channel=None,msg=None,log_level=0):
    global debug_messages
    return [d for d in debug_messages[::-1] if _filter_dbg_txt(channel,d[0]) and _filter_dbg_txt(msg,d[1]) and log_level <= d[2]]

def override_copy(v,actually_do_it=True):
    #return v
    from copy import deepcopy
    import copy
    # Hopefully we won't need this.
    # If we inject functions into the variables, it is possible we get Pickle Errors on compile time!
    #def new_copy(self, name=None):
    #    """Return a symbolic copy and optionally assign a name.
    #    Does not copy the tags.
    #    Also copies convis specific attributes.
    #    """
    #    global convis_attributes
    #    copied_variable = theano.tensor.basic.tensor_copy(self)
    #    copied_variable.name = name
    #    if hasattr(self,'preserve_labels_on_copy'):
    #        for a in convis_attributes:
    #            if hasattr(self,a):
    #                setattr(copied_variable,a,getattr(self,a))
    #        copied_variable.copied_from = self.v
    #    override_copy(copied_variable)
    #    return copied_variable
    def type_call(self,*args,**kwargs):
        new_v = self.__old_call__(*args,**kwargs)
        if hasattr(self,'v') and hasattr(self.v,'preserve_labels_on_copy'):
            for a in convis_attributes:
                if hasattr(self.v,a):
                    setattr(new_v,a,getattr(self.v,a))
            new_v.copied_from = self.v
        override_copy(new_v)
        return new_v
    def type_make_variable(self,name=None):
        new_v = self.Variable(self,name=name)
        if hasattr(self,'v') and hasattr(self.v,'preserve_labels_on_copy'):
            for a in convis_attributes:
                if hasattr(self.v,a):
                    setattr(new_v,a,getattr(self.v,a))
            new_v.copied_from = self.v
        override_copy(new_v)
        return new_v
    #v.copy = new.instancemethod(new_copy, v, None)
    v._old_type = v.type
    v.type = copy.copy(v.type)
    if not hasattr(v.type,'__old_call__'):
        v.type.__old_call__ = v.type.__call__
        v.type = copy.copy(v.type)
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
    v.__is_convis_var = True
    v.variable_type = 'state'
    if out_state is not None:
        v.state_out_state = out_state
    if init is not None:
        v.state_init = init
    if name is not None:
        v.name = name
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)
def as_out_state(v,in_state=None,init=None,name=None,**kwargs):
    v.__is_convis_var = True
    v.variable_type = 'out_state'
    if in_state is not None:
        as_state(in_state,out_state=v,init=init)
        v.state_in_state = in_state
    if name is not None:
        v.name = name
    if v.name is None:
        if in_state is not None and in_state.name is not None:
            v.name = in_state.name+'_out_state' 
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)

def as_input(v,name=None,**kwargs):
    v.__is_convis_var = True
    v.variable_type = 'input'
    if name is not None:
        v.name = name
    v.__dict__.update(kwargs)
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)
def as_variable(v,name=None,**kwargs):
    v.__is_convis_var = True
    if not hasattr(v,'variable_type'):
        # don't overwrite the variable type!
        v.variable_type = 'variable'
    if name is not None:
        v.name = name
    v.__dict__.update(kwargs)
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)
def as_output(v,name=None,**kwargs):
    v.__is_convis_var = True
    if not hasattr(v,'variable_type'):
        # is the output variable type even important?
        # in any case things can be eg. parameters and also outputs, so we don't want to overwrite this!
        v.variable_type = 'output'
    if name is not None:
        v.name = name
    v.__dict__.update(kwargs)
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)

def as_parameter(v,init=None,name=None,**kwargs):
    v.__is_convis_var = True
    v.variable_type = 'parameter'
    if init is not None:
        v.param_init = init
    if name is not None:
        v.name = name
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
            if not force and (getattr(self, 'initialized', False) or hasattr(self,'optimizing')):
                # The variable was already updated. To avoid cycles we do not update again.
                send_dbg('param.update',str(self.name)+' was already initialized or is optimizing itself.',0)
                return self
            # this might entail recursion when param_init calls update of other variables:
            new_val = self.param_init(create_context_O(self,update_trace=getattr(x,'update_trace',[])+[self]))
            #if self.name is not None and 'lambda' in self.name:
            #    print self.name, new_val
            send_dbg('param.update',str(self.name)+' updated itself from '+str(self.get_value())[:10]+' to '+str(new_val)[:10]+'.',2)
            self.set_value(new_val)
            self.initialized = True
            return self
        v.update = new.instancemethod(update, v, None)
    v.__dict__.update(kwargs)
    override_copy(v)
    return add_kwargs_to_v(v,**kwargs)


def find_a_class(class_name):
    class_path = class_name.split('.')
    try:
        class_reference = globals()[class_path[0]]
    except:
        class_reference = __import__(class_path[0])
        #locals()[class_path[0]]
    for i in range(1,len(class_path)):
        try:
            class_reference = getattr(class_reference,class_path[i])
        except:
            return None
    return class_reference
oid = 0
class O(object):
    """
        An `O` object is an object that allows easy access to its members.

        Example::

            o1 = O(a=1,b=2,c=3)
            print o1.a
            print dir(o1)

            print o1(d=4).d # creates a new O object with the added keywords

    """
    def __init__(self,**kwargs):
        global oid
        self._id=oid
        oid += 1
        self.__dict__.update(**dict([(save_name(k),v) for (k,v) in kwargs.items()]))
    def __call__(self,**kwargs):
        self.__dict__.update(**dict([(save_name(k),v) for (k,v) in kwargs.items()]))
        return self
    def __repr__(self):
        return 'Choices: '+(', '.join(self.__iterkeys__()))
    def _repr_html_(self):
        return repr(self)
    def __len__(self):
        return len([k for k in self.__dict__.keys() if not k.startswith('_')])
    def __iter__(self):
        return iter([v for (k,v) in self.__dict__.items() if not k.startswith('_')])
    def __iterkeys__(self):
        return iter([k for (k,v) in self.__dict__.items() if not k.startswith('_')])
    def __iteritems__(self):
        return iter([(k,v) for (k,v) in self.__dict__.items() if not k.startswith('_')])
    def __setattr__(self, name, value):
        if name in self.__dict__.keys() and hasattr(getattr(self, name),'set_value'):
            #print 'has set value'
            getattr(self, name).set_value(value)
        else:
            if name in self.__dict__.keys():
                #print 'setting in dictionary'
                self.__dict__[name] = value
            else:
                #print 'setting for object'
                object.__setattr__(self, name, value)
    def __setitem__(self,k,v):
        setattr(self,k,v)
    def __getitem__(self,k):
        if k in self.__dict__.keys():
            return getattr(self, k)
        if save_name(k) in self.__dict__.keys():
            return getattr(self, save_name(k))
        if str(k) != save_name(k):
            raise IndexError('Key not found: '+str(k)+' / '+save_name(k))
        raise IndexError('Key not found: '+str(k))
    @property
    def _(self):
        t = self.__dict__.get('_original_type',None)
        return self._get_as(t,recursive=True)
    def _get_as(self,as_type,recursive=False):
        if as_type == dict or as_type == 'dict':
            return dict([(k, getattr(v,'_', v) if recursive else v) for (k,v) in self.__iteritems__()])
        if as_type == list or as_type == 'list':
            import re
            keys = filter(lambda x: re.match(r'^n\d+$',x) != None, self.__dict__.keys())
            sorted_keys = sorted(map(lambda x: (len(x),x), keys))
            l = [self[k[1]] for k in sorted_keys]
            return [getattr(i,'_',i) if recursive else i for i in l]
        if type(as_type) is str:
            class_reference = find_a_class(as_type)
            if class_reference is not None:
                if 'config' in class_reference.__init__.im_func.func_code.co_varnames:
                    return class_reference(config=self._get_as(dict,recursive=True))
                else:
                    return class_reference(**self._get_as(dict,recursive=True))
        try:
            return as_type(**self._get_as(dict))
        except:
            pass
        raise Exception('type not recognized: '+str(as_type)+'')
        return self.__dict__.get('_item',None)

class _Search(O):
    def __init__(self,**kwargs):
        self._things = kwargs
    def __getattr__(self,search_string):
        return O(**dict([(save_name(k),v) for (k,v) in self._things.items() if search_string in k]))
    def __repr__(self):
        return 'Choices: enter a search term, enter a dot and use autocomplete to see matching items.'

class Ox(O):
    """
        An `Ox` object is an extended  O object that allows easy access to its members
        and automatically converts a hierarchical dictionaries and lists into a hierarchical `Ox.

        The special attributes `._all` and `._search` provide access to the flattend dictionary.

        Names will be converted to save variable names. Spaces and special characters are replaced with '_'.
        Lists will be replaced by dicitonaries with keys n0...nX according to the length of the list.

        To convert it back into dicitonaries and lists (and other obejcts that can be handled),
        use a single underscore as attribute.

        Example::

            o1 = Ox(**{'Node a':{'Subnode b':{'Number c':23},'Number d':24},'Node e':{'Float Value f':0.0}})
            
            ## Hierarchical
            # each level provides tab completion
            print o1.Node_a.Subnode_b.Number_c
            # prints 23
    
            ## Flattend
            print o1._all
            # prints: 'Choices: Node_a_Subnode_b_Number_c, Node_e_Float_Value_f, Node_a_Number_d'
            print o1._search.Number
            # prints: 'Choices: Node_a_Subnode_b_Number_c, Node_a_Number_d'

            ## as dictionary
            print o1.Node_a._
            # prints: {'Subnode b':{'Number c':23},'Number d':24}

        Searching:

            Using the special attribute '._search', a search string can be entered as a fake attribute.
            Entering a dot enables tab completion of all (flattend) entries that match the search string::

                o1.Node_a._search.Number.<Tab>
                # Will offer Subnode_b_Number_c and Number_d as completions


    """
    def __init__(self,**kwargs):
        for k,v in kwargs.iteritems():
            if not k.startswith('_'):
                self.__dict__[save_name(k)]= self.__make_Ox(v)
            else:
                self.__dict__[k] = v
        #super(Ox,self).__init__(**kwargs)
        if not hasattr(self,'_item'):
            self.__dict__['_item'] = kwargs
        if not hasattr(self,'_original_type'):
            self.__dict__['_original_type']='dict'
    def __make_Ox(self,thing):
        if type(thing) is Ox:
            return thing
        if type(thing) is dict:
            return Ox(**thing)
        if type(thing) is list:
            return Ox(_item=thing,_original_type='list',**dict([('n'+str(i),v) for i,v in enumerate(thing)]))
        if hasattr(thing,'as_ox_dict'):
            return Ox(_item=thing,_original_type=str(thing.type),**thing.as_ox_dict())
        return thing
    def __setattr__(self, name, value):
        if not name.startswith('_'):
            value = self.__make_Ox(value)
        super(Ox,self).__setattr__(name,value)
    def __getattribute__(self,key):
        if key.startswith('_'):
            return super(Ox,self).__getattribute__(key)
        return self.__make_Ox(self.__dict__[key])
    def _flatten(self,sep='_'):
        def flatten_rec(d):
            if hasattr(d,'__iteritems__'):
                return [(save_name(k if k1 is None else k+sep+k1),v1) for (k,kv) in  d.__iteritems__() for k1,v1 in flatten_rec(kv)]
            elif type(d) is dict:
                return [(save_name(k if k1 is None else k+sep+k1),v1) for (k,kv) in  d.items() for k1,v1 in flatten_rec(kv)]
            elif type(d) is list:
                return [(save_name('n'+str(i) if k1 is None else 'n'+str(i)+sep+k1),v1) for (i,kv) in  enumerate(d) for k1,v1 in flatten_rec(kv)]
            return [(None, d)]
        return dict(flatten_rec(self.__dict__))
    @property
    def _dict(self):
        return self.__dict__
    @property
    def _all(self):
        return O(**self._flatten())
    @property
    def _search(self):
        return _Search(**self._flatten())
    def __iter__(self):
        return iter([self.__make_Ox(v) for (k,v) in self.__dict__.items() if not k.startswith('_')])
    def __iteritems__(self):
        return iter([(k,self.__make_Ox(v)) for (k,v) in self.__dict__.items() if not k.startswith('_')])

class old_reverse_Ox(O):
    """
        An `Ox` object is an extended  O object that allows easy access to its members
        and automatically converts a hierarchical dictionaries and lists into a hierarchical `Ox.

        The special attributes `._all` and `._search` provide access to the flattend dictionary.

        Names will be converted to save variable names. Spaces and special characters are replaced with '_'.
        Lists will be replaced by dicitonaries with keys n0...nX according to the length of the list.

        Example::

            o1 = Ox(**{'Node a':{'Subnode b':{'Number c':23},'Number d':24},'Node e':{'Float Value f':0.0}})
            
            ## Hierarchical
            # each level provides tab completion
            print o1.Node_a.Subnode_b.Number_c
            # prints 23
    
            ## Flattend
            print o1._all
            # prints: 'Choices: Node_a_Subnode_b_Number_c, Node_e_Float_Value_f, Node_a_Number_d'
            print o1._search.Number
            # prints: 'Choices: Node_a_Subnode_b_Number_c, Node_a_Number_d'

        Searching:

            Using the special attribute '._search', a search string can be entered as a fake attribute.
            Entering a dot enables tab completion of all (flattend) entries that match the search string::

                o1.Node_a._search.Number.<Tab>
                # Will offer Subnode_b_Number_c and Number_d as completions


    """
    def __init__(self,**kwargs):
        super(Ox,self).__init__(**kwargs)
        if not hasattr(self,'_item'):
            self.__dict__['_item'] = kwargs
        if not hasattr(self,'_original_type'):
            self.__dict__['_original_type']='dict'
    def __make_Ox(self,thing):
        if type(thing) is dict:
            return Ox(**thing)
        if type(thing) is list:
            return Ox(_item=thing,_original_type='list',**dict([('n'+str(i),v) for i,v in enumerate(thing)]))
        return thing
    def __getattribute__(self,key):
        if key.startswith('_'):
            return super(Ox,self).__getattribute__(key)
        #if type(self.__dict__[key]) is dict:
        #    return Ox(_item=self.__dict__[key],**self.__dict__[key])
        #if type(self.__dict__[key]) is list:
        #    return Ox(_item=self.__dict__[key],**dict([('n'+str(i),v) for i,v in enumerate(self.__dict__[key])]))
        #    #return Ox(_item=self.__dict__[key],**dict([(save_name(k),v) for (k,v) in self.__dict__[key].items()]))
        return self.__make_Ox(self.__dict__[key])
    def _flatten(self,sep='_'):
        def flatten_rec(d):
            if type(d) is dict:
                return [(save_name(k if k1 is None else k+sep+k1),v1) for (k,kv) in  d.items() for k1,v1 in flatten_rec(kv)]
            if type(d) is list:
                return [(save_name('n'+str(i) if k1 is None else 'n'+str(i)+sep+k1),v1) for (i,kv) in  enumerate(d) for k1,v1 in flatten_rec(kv)]
            return [(None, d)]
        return dict(flatten_rec(self.__dict__))
    @property
    def _dict(self):
        return self.__dict__
    @property
    def _all(self):
        return O(**self._flatten())
    @property
    def _search(self):
        return _Search(**self._flatten())
    def __iter__(self):
        return iter([self.__make_Ox(v) for (k,v) in self.__dict__.items() if not k.startswith('_')])
    def __iteritems__(self):
        return iter([(k,self.__make_Ox(v)) for (k,v) in self.__dict__.items() if not k.startswith('_')])


def raise_exception(e):
    raise e
def create_context_O(var, **kwargs):
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
    if hasattr(var, 'config_key') and hasattr(var,'config_default'):
        return O(var=var,node=var.node,model=var.node.get_model(),
                 get_config=var.node.get_config,
                 value_from_config=lambda: var.node.get_config(var.config_key,var.config_default),
                 value_to_config=lambda v: var.node.set_config(var.config_key,v))(**kwargs)
    return O(var=var,node=var.node,model=var.node.get_model(),get_config=var.node.get_config,
             value_from_config=lambda: raise_exception(Exception('No config key and default value available. '+str(var.name)+'\n'+str(var.__dict__))))(**kwargs)

def create_hierarchical_O(vs,pi=0):
    """
        Creates an object that hierarchically represents the supplied items.
        The object is an interactively usable dictionary (see `O` objects) which provides tab completion for all entries.

        Each item is required to have a 'path' attribute which is a list of strings.
    """
    o = O()
    paths = f7([v.path[pi].name for v in vs if hasattr(v,'path') and len(v.path) > pi+1])
    leaves = f7([v for v in vs if hasattr(v,'path') and len(v.path) == pi+1])
    for p in paths:
        o(**{p: create_hierarchical_O([v for v in vs if hasattr(v,'path') and len(v.path) > pi and v.path[pi].name == p], pi+1)})
    for l in leaves:
        o(**{l.name: l})
    return o

def create_hierarchical_dict(vs,pi=0,name_sanitizer=save_name):
    """

    """
    o = {}
    paths = f7([name_sanitizer(v.path[pi].name) for v in vs if hasattr(v,'path') and len(v.path) > pi+1])
    leaves = f7([v for v in vs if hasattr(v,'path') and len(v.path) == pi+1])
    for p in paths:
        o.update(**{p: create_hierarchical_dict([v for v in vs if hasattr(v,'path') and len(v.path) > pi and name_sanitizer(v.path[pi].name) == p], pi+1)})
    for l in leaves:
        o.update(**{name_sanitizer(l.name): l})
    return o

def create_hierarchical_Ox(vs,pi=0):
    return Ox(**create_hierarchical_dict(vs,pi))

def create_hierarchical_dict_with_nodes(vs,pi=0,name_sanitizer=save_name):
    """
        name_sanitizer: eg. convis.base.save_name or str
    """
    o = {}
    paths = f7([v.path[pi] for v in vs if hasattr(v,'path') and len(v.path) > pi+1])
    leaves = f7([v for v in vs if hasattr(v,'path') and len(v.path) == pi+1])
    for p in paths:
        o.update(**{p: create_hierarchical_dict_with_nodes([v for v in vs if hasattr(v,'path') and len(v.path) > pi and v.path[pi] == p], pi+1)})
    for l in leaves:
        o.update(**{name_sanitizer(l.name): l})
    return o

def is_var(v):
    return hasattr(v,'__is_convis_var')
def is_state(v):
    return hasattr(v,'variable_type') and v.variable_type == 'state' and hasattr(v,'state_out_state')
def is_out_state(v):
    return hasattr(v,'variable_type') and v.variable_type == 'out_state'
def is_input_parameter(v):
    return hasattr(v,'variable_type') and (not hasattr(v,'get_value')) and v.variable_type == 'parameter'  and (not hasattr(v,'copied_from'))
def is_shared_parameter(v):
    return hasattr(v,'variable_type') and hasattr(v,'get_value') and v.variable_type == 'parameter'
def is_parameter(v):
    return is_input_parameter(v) or is_shared_parameter(v)
def is_input(v):
    return hasattr(v,'variable_type') and v.variable_type == 'input'
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

def shared_parameter(fun,init_object=O(),**kwargs):
    return as_parameter(theano.shared(fun(init_object)),
                        initialized = True,
                        init=fun,**kwargs)

def pad5(ar,N,axis=3,mode='mirror',c=0.0):
    """
        Padds a 5 dimensional tensor with `N` additional values.
        If the tensor has less than 5 dimensions, it will be extended.
        Returns a 5 dimensional tensor.

        Axis 0 and 2 are ignored.

        Usage:

            pad5 padds one axis at a time with one of the following modes:

            mode = 'mirror' (default)
                the image is mirrored at the edges, such that image statistics are similar

            mode = 'border'
                the border pixels are repeated 

            mode = 'const'
                the padded area is filled with a single value (default 0.0)
                It can also be a theano varible such as the mean of the tensor that is to be padded.

        For axis 1 (time), the padding happens exclusively at the front,
        for axes 3 and 4 (x and y) the amount is split into `N/2` and `N-(N/2)` (this can be asymmetric!).
        The total amount padded is always `N`.

        For convenience, `pad5_txy` can pad time, x and y with the same mode simultaneously.
        `pad3` and `pad2` return 3 tensors and matrices after padding.
    """
    ar = theano_utils.make_nd(ar,5)
    if N == 0:
        return ar
    N1,N2 = (N/2),N-(N/2)
    if mode == 'mirror':
        if axis == 1:
            return T.concatenate([ar[:,N:0:-1,:,:,:],ar],axis=1)
        if axis == 3:
            return T.concatenate([ar[:,:,:,N1:0:-1,:],ar,ar[:,:,:,-1:-(N2+1):-1,:]],axis=3)
        if axis ==4:
            return T.concatenate([ar[:,:,:,:,N1:0:-1],ar,ar[:,:,:,:,-1:-(N2+1):-1]],axis=4)
    if mode == 'border':
        if axis == 1:
            return T.concatenate([ar[:,:1,:,:,:]]*N+[ar],axis=1)
        if axis == 3:
            return T.concatenate([ar[:,:,:,:1,:]]*N1+[ar]+[ar[:,:,:,-1:,:]]*N2,axis=3)
        if axis ==4:
            return T.concatenate([ar[:,:,:,:,:1]]*N1+[ar]+[ar[:,:,:,:,-1:]]*N2,axis=4)
    if mode == 'const':
        if axis == 1:
            return T.concatenate([c*T.ones_like(ar[:,N:0:-1,:,:,:]),ar],axis=1)
        if axis == 3:
            return T.concatenate([c*T.ones_like(ar[:,:,:,N1:0:-1,:]),ar,c*T.ones_like(ar[:,:,:,-1:-(N2+1):-1,:])],axis=3)
        if axis ==4:
            return T.concatenate([c*T.ones_like(ar[:,:,:,:,N1:0:-1]),ar,c*T.ones_like(ar[:,:,:,:,-1:-(N2+1):-1])],axis=4)
        
def pad5_txy(ar,Nt,Nx,Ny,mode='mirror',c=0.0):
    """
        padds a 5 dimensional tensor with additional values

            Nt: number of steps added to the temporal dimension
            Nx,Ny: number of pixels added to the spatial dimensions

        see `pad5` for mode and c
    """
    return pad5(pad5(pad5(ar,Nt,1,mode=mode),Nx,3,mode=mode),Ny,4,mode=mode)

def pad3(ar,N,axis=3,mode='mirror',c=0.0):
    """
        Padds a 3 dimensional tensor at axis `axis` with `N` bins.
        If the tensor does not have 3 dimensions, it will be converted.
        Returns a 3 dimensional tensor.

        see `pad5`
    """
    if axis in [0,1,2]:
        axis = {0:1, 1:3, 2:4}[axis]
    else:
        raise Exception('pad3 only accepts axis 0,1,2! Use pad5 for higer dimension tensors')
    return pad5(theano_utils.make_nd(theano_utils.make_nd(ar,5),N=N,axis=axis,mode=mode,c=c),3)
def pad3_txy(ar,Nt,Nx,Ny,mode='mirror',c=0.0):
    """
        Padds a 3 dimensional tensor with `Nt` bins in time and `Nx` and `Ny` bins in x and y direction.
        If the tensor does not have 3 dimensions, it will be converted.
        Returns a 3 dimensional tensor.

        see `pad5_txy` and `pad5`
    """
    return theano_utils.make_nd(pad5_txy(theano_utils.make_nd(ar,5),Nt,Nx,Ny,mode=mode,c=c),3)

def pad2(ar,N,axis=3,mode='mirror',c=0.0):
    """
        Padds a 2 dimensional tensor at axis `axis` with `N` bins.
        If the tensor does not have 2 dimensions, it will be converted.
        Returns a 2 dimensional tensor.

        see `pad5`
    """
    if axis in [0,1]:
        axis = {0:3, 1:4}[axis]
    else:
        raise Exception('pad2 only accepts axis 0 and 1! Use pad5 for higer dimension tensors')
    return theano_utils.make_nd(pad5(theano_utils.make_nd(ar,5),N=N,axis=axis,mode=mode,c=c),2)

def pad2_xy(ar,Nx,Ny,mode='mirror',c=0.0):
    """
        Padds a 2 dimensional tensor with `Nx` and `Ny` bins in x and y direction.
        If the tensor does not have 2 dimensions, it will be converted.
        Returns a 2 dimensional tensor.

        see `pad5_txy` and `pad5`
    """
    return theano_utils.make_nd(pad5_txy(theano_utils.make_nd(ar,5),0,Nx,Ny,mode=mode,c=c),2)

### Node and Model classes

def len_parents(n):
    if hasattr(n,'parent') and n.parent != n:
        return len_parents(n.parent)+1
    return 0

class GraphWrapper(object):
    """
        

    """
    parent = None
    config_dict = None
    node_type = 'Node'
    node_description = ''
    def __init__(self,graph,name,m=None,parent=None,ignore=[],scan_op=None,**kwargs):
        self.m = m
        self.parent = parent
        self.graph = graph
        if hasattr(self.graph,'root_of'):
            return self.graph.root_of
        if not hasattr(self.graph,'__is_convis_var'):
            self.graph = as_output(T.as_tensor_variable(self.graph))
        if self.graph.name is None:
            # we only replace the name if it is necessary
            self.graph.name = 'output'
        self.graph.root_of = self
        #self.outputs = [self.graph]
        self.name = name
        self.ignore = ignore
        self.scan_op = scan_op
        self.__dict__.update(kwargs)
        self.follow_scan = True
        if self.follow_scan and scan_op is None:
            self.wrap_scans(self.graph)
        self.label_variables(self.graph)
        #self.label_variables(self.scan_outputs,follow_scan=False) # just labeling these to prevent infinite recursion
        #self.inputs = theano_utils.get_input_variables_iter(self.graph)
        #if self.m is not None:
        #    self.m.add(self)
    def get_model(self):
        if hasattr(self,'model'):
            return self.model
        if self.parent is not None:
            return self.parent.get_model()
    def get_config(self,key=None,default=None,type_cast=None):
        if key is None:
            return self.config_dict
        if type_cast is not None:
            return type_cast(self.get_config(key,default))
        return self.config.get(key,default)
    def set_config(self,key,v=None):
        if v is None and hasattr(key,'get'):
            send_dbg('set_config',str(getattr(self,'name',''))+' replaced config: '+str(key)+'',1)
            self.config_dict = key
        else:
            send_dbg('set_config',str(getattr(self,'name',''))+' set config: '+str(key)+': '+str(v),1)
            self.config_dict[key] = v
    @property
    def config(self):
        if self.config_dict is None:
            if self.parent is not None:
                return self.parent.config
            raise Exception('GraphWrapper '+str(self.name)+' has no configuration! But also no parent!')
        return self.config_dict
    def get_parents(self):
        p = []
        if self.parent is not None:
            p.append(self.parent)
            if hasattr(self.parent,'get_parents'):
                p.extend(self.parent.get_parents())
        return p
    def wrap_scans(self,g):
        my_scan_vars = filter(lambda x: theano_utils.is_scan_op(x), theano_utils.get_variables_iter(g,ignore=self.ignore,explore_scan=False,include_copies=False))
        for i,ow in enumerate(f7([v.owner for v in my_scan_vars])):
            op = ow.op
            variables_leading_to_op = [v for v in my_scan_vars if v.owner.op is op]
            #print ow, variables_leading_to_op
            GraphWrapper(as_output(T.as_tensor_variable(op.outputs),name='scan_group_'+str(i)),scan_op=op,name='Scan Loop '+str(i),ignore=[self.graph]+self.ignore+ow.inputs) # creating a sub node
            #+ow.inputs
            self.label_variables([o for o in op.outputs])
    def label_variables(self,g,follow_scan=True,max_depth=2):
        if max_depth <= 0:
            return
        my_named_vars = theano_utils.get_named_variables_iter(g,ignore=self.ignore,explore_scan=True,include_copies=True)
        # variables that don't have a name are not tracked.
        # exception: we name and claim all scan op variables, since they mess up plotting the graph!
        my_scan_vars = filter(lambda x: theano_utils.is_scan_op(x), theano_utils.get_variables_iter(g,ignore=self.ignore,explore_scan=False,include_copies=False))
        if not follow_scan or not self.follow_scan:
            my_scan_vars = []
        scan_ops = {}
        for i,v in enumerate(my_scan_vars):
            if v.name is None:
                v.name = 'scan_output_'+str(i)
            as_output(v)
        for v in my_named_vars + [v for v in my_scan_vars if v.owner.op != self.scan_op]:
            if hasattr(v,'path'):
                if v.path[0] == self:
                    continue # we already are the owner of this variable
            if hasattr(v,'full_name'):
                v.full_name = self.name+'.'+v.full_name
                v.path = [self] + v.path
            else:
                v.full_name = self.name+'.'+v.name
                v.path = [self,v]
            global do_debug
            if do_debug:
                print 'labeled: ',v.path
            if hasattr(v,'node') and v.node != None and v.node != self:
                if v.node.parent is None:
                    v.node.parent = self
            else:
                v.node = self
            if not hasattr(v,'simple_name') or v.simple_name is None:
                v.simple_name = v.name
            #v.node = self
    def _as_TensorVariable(self):
        """
            When theano uses an object as a variable it will first check if it supports this function.
            We simply act as if we are the annotated graph that is stored inside.

            Note from theano source: # TODO: pass name and ndim arguments
        """
        return self.graph
    #@property
    #def inputs(self):
    #    return create_hierarchical_Ox(filter(is_input,theano_utils.get_named_variables_iter(self.graph)),pi=len_parents(self))
    #@property
    #def outputs(self):
    #    return [self.graph]
    def filter_variables(self,filter_func=is_input):
        return create_hierarchical_Ox(filter(filter_func,theano_utils.get_named_variables_iter(self.graph)),pi=len_parents(self))
    @property
    def output(self):
        return self.graph
        #return create_hierarchical_O(theano_utils.get_input_variables_iter(self.graph),pi=1)
    @property
    def parameters(self):
        return create_hierarchical_Ox(filter(is_shared_parameter,theano_utils.get_named_variables_iter(self.graph)),pi=len_parents(self))
    @property
    def params(self):
        return create_hierarchical_Ox(filter(is_shared_parameter,theano_utils.get_named_variables_iter(self.graph)),pi=len_parents(self))
    @property
    def states(self):
        return create_hierarchical_Ox(filter(is_state,theano_utils.get_named_variables_iter(self.graph)),pi=len_parents(self))
    @property
    def variables(self):
        return create_hierarchical_Ox(theano_utils.get_named_variables_iter(self.graph),pi=len_parents(self))
    def __repr__(self):
        if hasattr(self, 'node_description') and callable(self.node_description):
            # we can provide a dynamic description
            return '['+str(self.node_type)+'] ' + self.name + ': ' + str(self.node_description())
        return '['+str(self.node_type)+'] ' + self.name + ': ' + str(self.node_description)
    def map_state(self,in_state,out_state,**kwargs):
        as_state(in_state,out_state=out_state,**kwargsW)
    def replace(self,a,b):
        """todo: decide of whether to do this here or in node or model (or different alltogether)"""
        for o in self.graph:
            theano_utils._replace(o,b,a)
        # replace in out states
        for o in [v.state_out_state for v in filter(is_state,self.get_variables())]:
            theano_utils._replace(o,b,a)
    def shared_parameter(self, f=lambda x:x, name='',**kwargs):
        # todo: where to save config?
        if 'config_key' in kwargs.keys() and 'config_default' in kwargs.keys():
            return shared_parameter(f,
                                    O()(node=self,
                                        model=self.model,
                                        get_config=self.get_config,
                                        value_from_config=lambda: self.get_config(kwargs.get('config_key'),kwargs.get('config_default')),
                                        value_to_config=lambda v: self.set_config(kwargs.get('config_key'),v)),
                                    name=name,**kwargs)
        return shared_parameter(f,O()(node=self,model=self.model,get_config=self.get_config),name=name,**kwargs)
    def add_input(self,other,replace_inputs=do_replace_inputs,input=None):
        if input is None:
            if hasattr(self, 'default_input'):
                input = self.default_input
            elif hasattr(self.variables, 'input'):
                input = self.variables.input
            else:
                raise Exception('No input found in '+getattr(self,'name','[unnamed node]')+'!')
        if type(input) is str:
            if input in self.inputs.keys():
                input = self.inputs[input]
            else:
                raise Exception('Input "'+str(input)+'"" found in '+getattr(self,'name','[unnamed node]')+' inputs!')
        if do_debug:
            print 'connecting',other.name,'to',self.name,''
        v = other
        if hasattr(other,'_as_TensorVariable'):
            v = other._as_TensorVariable()
        if type(input.owner.op) == T.elemwise.Sum:
            if do_debug:
                print 'Adding to sum'
            # assuming a 3d input/ output
            if replace_inputs and hasattr(input.owner.inputs[0].owner.inputs[1].owner.inputs[0],'replaceable_input'):
                input.owner.inputs[0].owner.inputs[1] = v.dimshuffle(('x',0,1,2))
            else:
                input.owner.inputs[0].owner.inputs.append(v.dimshuffle(('x',0,1,2)))
            if do_debug:
                print 'inputs are now:',input.owner.inputs[0].owner.inputs
            if not hasattr(v, 'connects'):
                v.connects = []
            v.connects.append([self,other])
        else:
            if hasattr(input,'variable_type') and input.variable_type == 'input':
                input.variable_type = 'replaced_input'
            if not hasattr(v, 'connects'):
                v.connects = []
            v.connects.append([self,other])
            try:
                theano_utils._replace(self.output,input,v)
            except:
                print 'Something not found! ',other.variables,self.variables
    def __iadd__(self,other):
        self.add_input(other)
        return self
    def __neg__(self):
        return -self.graph
    def __add__(self,other):
        return self.graph + other
    def __radd__(self,other):
        return other + self.graph
    def __mul__(self,other):
        return self.graph * other
    def __rmul__(self,other):
        return other * self.graph
    def __sub__(self,other):
        return self.graph - other
    def __rsub__(self,other):
        return other - self.graph
    def __div__(self,other):
        return self.graph / other
    def __rdiv__(self,other):
        return other / self.graph
    def __floordiv__(self,other):
        return self.graph // other

class N(GraphWrapper):
    #parameters = []
    states = {}
    state_initializers = {}
    inputs = OrderedDict()
    def __init__(self,graph,name=None,m=None,parent=None,config=None,**kwargs):
        if name is None:
            name = str(uuid.uuid4())
        self.m = m
        self.parent = parent
        # this input can be used in the graph or simply ignored. However, then automatic connections no longer work!
        #self.graph = T.as_tensor_variable(graph)
        #if self.graph.name is None:
        #    self.graph.name = 'output'
        #self.outputs = [self.graph]
        #self.name = name
        #self.set_name(name)
        #self.inputs = theano_utils.get_input_variables_iter(self.graph)
        self.node_type = 'Node'
        self.node_description = ''
        if config is not None:
            self.set_config(config)
        if self.m is not None:
            self.m.add(self)
        if self.config is None:
            raise Exception('No config for node '+str(self.name)+'! Use .set_config({}) before calling super constructor!')
        if not hasattr(self,'default_input'):
            raise Exception('No input defined for node '+str(self.name)+'! Use .create_input(...) before calling super constructor!')
        super(N, self).__init__(graph,name=name,m=m,parent=parent)
    def create_input(self,n=1,name='input',sep='_'):
        if n == 1:
            self.input = T.sum([as_input(T.dtensor3(),name,replaceable_input=True)],axis=0)
            self.default_input = self.input
            self.inputs.update(OrderedDict([(name,self.input)]))
            return self.input
        elif type(n) == int:
            self.inputs.update(OrderedDict([(name+sep+str(i),T.sum([as_input(T.dtensor3(),name+sep+str(i),replaceable_input=True)],axis=0)) for i in range(n)]))
            self.default_input = self.inputs.values()[0]
            self.input = T.join(*([0]+self.inputs.values()))
            return self.inputs.values()
        elif type(n) in [list,tuple]:
            self.inputs.update(OrderedDict([(input_name,T.sum([as_input(T.dtensor3(),str(input_name),replaceable_input=True)],axis=0)) for input_name in n]))
            self.default_input = self.inputs[n[0]]
            self.input = self.inputs[n[0]]
            return self.inputs
        else:
            raise Exception('Argument not understood. Options are: an int (either 1 for a single input or >1 for more) or a list of names for the inputs.')
    def get_parents(self):
        p = []
        if self.parent is not None:
            p.append(self.parent)
            if hasattr(self.parent,'get_parents'):
                p.extend(self.parent.get_parents())
        return p
    def set_name(self,name):
        pass
        #self.name = name
        #my_named_vars = theano_utils.get_named_variables_iter(self.graph)
        #for v in my_named_vars:
        #    if hasattr(v,'node') and v.node != self:
        #        continue
        #        #raise Exception('Variable registered twice!')
        #    if not hasattr(v,'simple_name') or v.simple_name is None:
        #        v.simple_name = v.name
        #    v.node = self
        #    v.name = self.name+'_'+v.simple_name
        #self.variables = dict([(v.simple_name,v) for v in my_named_vars])
    def get_variables(self):
        my_named_vars = theano_utils.get_named_variables_iter(self.graph)
        return dict([(v,v) for v in my_named_vars])
    def get_input_variables(self):
        return dict([(v,self) for v in theano_utils.get_input_variables_iter(self.graph)])
    def get_output_variables(self):
        return {self.graph: self}
    def set_m(self,m):
        self.m = m
        return self
    def var(self,v):
        raise Exception('\'%s\' not a variable of this node'%v)
    def map_state(self,in_state,out_state,**kwargs):
        self.states[in_state] = out_state
        as_state(in_state,out_state=out_state,**kwargsW)
    def replace(self,a,b):
        theano_utils._replace(self.graph,b,a)
        # replace in out states
        for o in [v.state_out_state for v in filter(is_state,self.get_variables())]:
            theano_utils._replace(o,b,a)
    def get_config(self,key,default=None,type_cast=None):
        if type_cast is not None:
            return type_cast(self.get_config(key,default))
        if not hasattr(self,'config'):
            return default
        return self.config.get(key,default)
        #def set_config(self,key,v=None):
        #if v is None and hasattr(key,'get'):
        #    super(N, self).set_config(key)
        #else:
        #    self.config[key] = v
        #notify model? self.model.set_config(self,key,v)
    def shared_parameter(self, f=lambda x:x, name='',**kwargs):
        if 'config_key' in kwargs.keys() and 'config_default' in kwargs.keys():
            return shared_parameter(f,
                                    O()(node=self,
                                        model=self.model,
                                        get_config=self.get_config,
                                        value_from_config=lambda: self.get_config(kwargs.get('config_key'),kwargs.get('config_default')),
                                        value_to_config=lambda v: self.set_config(kwargs.get('config_key'),v)),
                                    name=name,**kwargs)
        return shared_parameter(f,O()(node=self,model=self.model,get_config=self.get_config),name=name,**kwargs)
    def shape(self,input_shape):
        # unless this node does something special, the shape of the output should be identical to the input
        return input_shape
    def __add__(self,other):
        raise Exception('Not implemented!')
        ##
        # to get this working, I need to have a sure way to clone nodes with all the associated info in the graph.
        # after that this function should merge two nodes and return a new one
        # this means that it's not really for connecting two nodes that we already have in a model, but rather replacing them
        # with a new one.
        return N()

class _Search(O):
    def __init__(self,**kwargs):
        self._things = kwargs
    def __getattr__(self,search_string):
        return O(**dict([(save_name(k),v) for (k,v) in self._things.items() if search_string in k]))
    def __repr__(self):
        return 'Choices: enter a search term, enter with a dot and use autocomplete to see matching items.'

class _Vars(O):
    def __init__(self,model,**kwargs):
        self._model = model
        super(_Vars,self).__init__(**kwargs)
        vars = [v for v in self._model.get_variables()  if v.name is not None]
        nodes = f7([v.node for v in vars if hasattr(v,'node')])
        for n in nodes:
            self.__dict__[save_name(n.name)] = O(**dict([(save_name(k.simple_name),k) for k in vars if hasattr(k,'node') and k.node == n]))
        self.__dict__['_all'] = O(**dict([(full_path(k),k) for k in vars if hasattr(k,'path')]))
        self.__dict__['_search'] = _Search(**dict([(full_path(k),k) for k in vars if hasattr(k,'path')]))

class _Configuration(O):
    def __init__(self,model,**kwargs):
        self._model = model
        super(_Configuration,self).__init__(**kwargs)
        vars = [v for v in self._model.get_variables() if is_parameter(v)]
        nodes = f7([v.node for v in vars if hasattr(v,'node')])
        for n in nodes:
            self.__dict__[save_name(n.name)] = O(**dict([(str(k.simple_name),k) for k in vars  if hasattr(k,'node') and k.node == n]))

def connect(list_of_lists):
    """
        Connects nodes alternatingly in sequence and parallel:

        [A,B,[[C,D],[E,F],G] will results in two paths:

            A -> B -> C -> D -> G
            A -> B -> E -> F -> G


    """
    def connect_in_sequence(l,last_outputs = []):
        if not type(l) is list:
            return [[l,l]]
        last_outputs = []
        for e in l:
            new_outputs = connect_in_parallel(e,last_outputs=last_outputs)
            if len(last_outputs) > 0 and len(new_outputs) > 0:
                for e1 in last_outputs:
                    for e2 in new_outputs:
                        e2[0]+=e1[1]
            last_outputs = new_outputs
        return [l[0],l[-1]]
    def connect_in_parallel(l,last_outputs=[]):
        if not type(l) is list:
            return [[l,l]]
        last_elements = []
        for e in l:
            last_elements.append(connect_in_sequence(e,last_outputs))
        return last_elements
    connect_in_sequence(list_of_lists)
    return list_of_lists

class Output(object):
    def __init__(self,outs,keys=None):
        """
            This object provides a container for output numpy arrays which are labeled with theano variables.

            The outputs can be queried either by sorted order (like a simple list),
            by the theano variable which represents this output, the name of this variable
            or the full path of the variable.
            To make this meaningfull, provide a name to your output variables.

            In the case of name collisions, the behavior of OrderedDict will use the last variable added.

            The full path names of all variables are also added to this objects __dict__,
            allowing for tab completion.
        """
        self._out_dict = OrderedDict({})
        self._out_dict_by_full_names = OrderedDict({})
        self._out_dict_by_short_names = OrderedDict({})
        self._outs = outs
        self.keys = keys
        if keys is not None:
            self._out_dict = OrderedDict(zip(keys,outs))
            self._out_dict_by_full_names = OrderedDict([(full_path(k),o) for (k,o) in zip(keys,outs)])
            self._out_dict_by_short_names = OrderedDict([(save_name(k.name),o) for (k,o) in zip(keys,outs) if hasattr(k,'name') and type(k.name) is str])
        self.__dict__.update(self._out_dict_by_full_names)
    def __len__(self):
        return len(self._outs)
    def __iter__(self):
        return iter(self._outs)
    def __getitem__(self,k):
        if type(k) is int:
            return self._outs[k]
        else:
            if k in self._out_dict.keys():
                return self._out_dict[k]
            if save_name(k) in self._out_dict_by_short_names.keys():
                return self._out_dict_by_short_names[save_name(k)]
            if save_name(k) in self._out_dict_by_full_names.keys():
                return self._out_dict_by_full_names[save_name(k)]
        if str(k) != save_name(k):
            raise IndexError('Key not found: '+str(k)+' / '+save_name(k))
        raise IndexError('Key not found: '+str(k))


class M(object):
    def __init__(self, size=(10,10), pixel_per_degree=10.0, steps_per_second= 1000.0, **kwargs):
        self.debug = False
        self.nodes = []
        self.var_outputs = {}
        self.mappings = {}
        self.givens = {}
        self.outputs = []
        self.config = {}
        self.module_graph = []
        self.pixel_per_degree = pixel_per_degree
        self.steps_per_second = steps_per_second
        self.size_in_degree = size
        self.__dict__.update(kwargs)
    @property
    def shape(self):
        return tuple([np.newaxis]+[int(self.degree_to_pixel(x)) for x in self.size_in_degree])
    @shape.setter
    def shape(self,new_shape):
        self.size_in_degree = tuple(self.pixel_to_degree(x) for x in new_shape[1:])
    @property
    def c(self):
        return _Configuration(self)
    @property
    def v(self):
        return _Vars(self)
    @property
    def parameters(self):
        return create_hierarchical_Ox(filter(is_shared_parameter,theano_utils.get_named_variables_iter(self.outputs)),pi=len_parents(self))
    @property
    def states(self):
        return create_hierarchical_Ox(filter(is_state,theano_utils.get_named_variables_iter(self.outputs)),pi=len_parents(self))
    @property
    def variables(self):
        return create_hierarchical_Ox(theano_utils.get_named_variables_iter(self.outputs),pi=len_parents(self))
    def degree_to_pixel(self,degree):
        return float(degree) * self.pixel_per_degree
    def pixel_to_degree(self,pixel):
        return float(pixel) / self.pixel_per_degree
    def seconds_to_steps(self,t):
        return float(t) * self.steps_per_second
    def steps_to_seconds(self,steps):
        return float(steps) / self.steps_per_second
    def add(self,n):
        if n.name in map(lambda x: x.name, self.nodes):
            pass#raise Exception('Node named %s already exists!'%n.name)
        self.nodes.append(n.set_m(self))
        self.var_outputs.update(n.get_output_variables())
    def __contains__(self,item):
        raise Exception('Will be reimplemented!')
    def map(self,a,b):
        raise Exception('Will be reimplemented!')
    def out(self,a):
        """ 
            Will add a variable as an output.

            If this function is provided with a node, it tries to add an attribute `output`,
            then add a list in the attribute `outputs`, then a variable named `output`.
        """
        if hasattr(a,'node'):
            self.add(a.node)
            self.outputs.append(a)
        else:
            self.add(a)
            if hasattr(a,'output'):
                self.outputs.append(getattr(a,'output'))
            elif hasattr(a,'outputs'):
                self.outputs.extend(getattr(a,'outputs'))
            else:
                self.outputs.append(a.var('output'))
    def add_output(self,a,name=None):
        if hasattr(a,'graph'):
            a = a.graph
        if getattr(a,'name',None) is None and name is not None:
            a = as_output(a)
            a.name = name
        self.outputs.append(a)
    def in_out(self,a,b):
        #self.map(b.var('input'),a.var('output'))
        if hasattr(a,'graph'):
            a = a.graph
        print 'Replacing:',a,b
        if issubclass(b.__class__, N):
            if hasattr(b.var('input'),'variable_type') and b.var('input').variable_type == 'input':
                b.var('input').variable_type = 'replaced_input'
            #theano_utils._replace(b.output,b.var('input'),a.var('output'))
            if not hasattr(a, 'connects'):
                a.connects = []
            a.connects.append([b,getattr(a,'node',a)])
            try:
                theano_utils._replace(b.output,b.variables.input,a)
            except:
                print 'Something not found! ',a,b.variables
        elif hasattr(b,'node'):
            if not hasattr(a, 'connects'):
                a.connects = []
            a.connects.append([b.node,getattr(a,'node',a)])
            theano_utils._replace(b.node.output,b,a)
        else:
            raise Exception('This is not a node and not a variable with a node. Maybe the variable was not named?')
        #self.module_graph.append([a,b]) # no longer used??
        #b.replace(b.var('input'),a.var('output'))
    def _in_out(self,a,b):
        #self.map(b.var('input'),a.var('output'))
        aa = a
        bb = b
        if issubclass(a.__class__, N):
            aa = a.output
            #aa = a.var('output')
        if issubclass(b.__class__, N):
            bb = b.var('input')
        if hasattr(bb,'variable_type') and bb.variable_type == 'input':
            bb.variable_type = 'replaced_input'
        for o in b.outputs:
            theano_utils.replace(o,bb,aa)
    def get_variables(self):
        vs = []
        for o in self.outputs:
            vs.extend(theano_utils.get_variables_iter(o))
        return f7(vs)
    def describe_variables(self):
        return [describe(v) for v in self.get_variables()]

    def _deprecated_get_inputs(self,outputs=None): 
        import copy
        if outputs is None:
            outputs = copy.copy(self.outputs)
        seen = []
        inputs = []
        while True:
            outputs = f7(outputs)
            #print outputs,seen,inputs
            if len(outputs) == 0:
                break
            o = output
            outputs.remove(o)
            if o in seen:
                continue
            seen.append(o)
            inputs_to_o = theano_utils.get_input_variables_iter(o)
            if o in self.mappings.keys():
                #print 'mapping for ',o
                outputs.append(self.mappings[o])
            elif o in self.var_outputs.keys():
                #print 'inout for ',o
                outputs.extend(self.var_outputs[o].inputs)
            elif inputs_to_o != [o]:
                #print 'inputs: ',inputs_to_o
                outputs.extend(theano_utils.get_input_variables_iter(o))
            else:
                #print "No substitution!"
                inputs.append(o)
        return inputs 
    def create_function(self,updates=None,additional_inputs=[]):
        if self.debug is not True:
            # we disable warnings from theano gof because of the unresolved cache leak
            # TODO: fix the leak and re-enable warnings
            import logging
            logging.getLogger("theano.gof.cmodule").setLevel(logging.ERROR) 
        if updates is None:
            updates = theano.updates.OrderedUpdates()
        for a,b in  self.mappings.items():
            for n in self.nodes:
                if a.variable_type == 'input':
                    a.variable_type = 'replaced_input'
                theano_utils.replace(n.graph,a,b)
        outputs = [o._as_TensorVariable() if hasattr(o,'_as_TensorVariable') else o for o in self.outputs]
        variables = f7([v for o in outputs for v in theano_utils.get_variables_iter(o)])
        for v in variables:
            if hasattr(v,'updates'):
                updates[v] = np.sum([u for u in v.updates])
        self.compute_input_order = f7(additional_inputs + filter(is_input,variables) + filter(is_input_parameter,variables))
        self.additional_inputs = additional_inputs
        self.compute_state_inits = []
        state_variables = filter(is_state,variables)
        self.compute_output_order = f7(outputs + [v.state_out_state for v in state_variables])
        self.compute_updates_order = theano.updates.OrderedUpdates()
        self.compute_updates_order.update(updates)
        for state_var in state_variables:
            self.compute_input_order.append(state_var)
            #self.compute_updates_order[state_var] = state_var.state_out_state
            #print state_var.state_out_state
            self.compute_state_inits.append(state_var.state_init)
        givens = [(a,b) for (a,b) in self.givens.items()]
        self.compute_input_dict = dict((v,None) for v in self.compute_input_order)
        self.compute_state_dict = dict((v,None) for v in self.compute_output_order if is_out_state(v))
        if self.debug:# hasattr(retina, 'debug') and retina.debug:
            print 'solving for:',outputs
            print 'all variables:',len(variables)
            print 'input:',filter(is_input,variables)
            print 'parameters:',filter(is_input_parameter,variables)
            print 'states:',state_variables
            print 'updates:',self.compute_updates_order
        self.reset_parameters()
        self.compute = theano.function(inputs=self.compute_input_order, 
                                    outputs=self.compute_output_order, 
                                    updates=self.compute_updates_order,
                                    givens=givens,on_unused_input='ignore')
        if self.debug is not True:
            import logging
            logging.getLogger("theano.gof.cmodule").setLevel(logging.WARNING) 
    def clear_states(self):
        self.compute_state_dict = {}
    def reset_parameters(self):
        for shared_parameter in self.parameters._all:
            if is_shared_parameter(shared_parameter) and hasattr(shared_parameter,'initialized'):
                shared_parameter.initialized = False
    def run_in_chunks(self,the_input,max_length,additional_inputs=[],inputs={},run_after=None,**kwargs):
        chunked_output = []
        t = 0
        while t < the_input.shape[0]:
            if self.debug:
                print 'Chunk:', t, t+max_length
            oo = self.run(the_input[t:(t+max_length)],
                          additional_inputs=[ai[t:(t+max_length)] for ai in additional_inputs],
                          inputs=dict([(k,i[t:(t+max_length)]) for k,i in inputs.items()]),
                          run_after=run_after)
            chunked_output.append([o for o in oo])
            t += max_length
        return Output(np.concatenate(chunked_output,axis=1),keys=self.compute_output_order[:len(self.outputs)])
    def run_in_chuncks(self,the_input,max_length,additional_inputs=[],inputs={},run_after=None,**kwargs):
        """typo"""
        return self.run_in_chunks(the_input,max_length,additional_inputs=additional_inputs,inputs=inputs,run_after=run_after,**kwargs)
    def run(self,the_input,additional_inputs=[],inputs={},run_after=None,**kwargs):
        if not hasattr(self,'compute_input_dict'):
            # todo: manage multiple functions
            self.create_function()
        c = O()
        c.input = the_input
        c.model = self
        #c.config = self.config
        input_dict = {}
        input_dict.update(dict(zip(self.additional_inputs,additional_inputs)))
        for k,v in self.compute_input_dict.items():
            if v is not None:
                if hasattr(v,'array'):
                    input_dict[k] = v
                elif hasattr(v,'get'):
                    input_dict[k] = v.get(len(the_input))
            else:
                if is_input(k):
                    if k not in self.additional_inputs:
                        if k in inputs.keys():
                            input_dict[k] = inputs[k]
                        else:
                            input_dict[k] = the_input
                if is_input_parameter(k):
                    input_dict[k] = k.param_init(create_context_O(k,input=the_input))
                if is_state(k):
                    if self.compute_state_dict.get(k.state_out_state,None) is None:
                        input_dict[k] = k.state_init(create_context_O(k,input=the_input))
                    else:
                        input_dict[k] = self.compute_state_dict[k.state_out_state]
        for shared_parameter in self.parameters._all:
            if is_shared_parameter(shared_parameter):
                #if (not hasattr(shared_parameter,'initialized') or shared_parameter.initialized == False) and not hasattr(shared_parameter,'optimized'):
                # until we can track which config values where changed, we re-initialize everything
                # all smart stuff is now in the injected .update method
                shared_parameter.update(create_context_O(shared_parameter,input=the_input))
                #shared_parameter.initialized = True
        the_vars = [input_dict[v] for v in self.compute_input_order]
        the_output = self.compute(*the_vars)
        if run_after is not None:
            run_after(the_output,self)
        for (o,k) in zip(the_output, self.compute_output_order):
            if is_out_state(k):
                self.compute_state_dict[k] = o
        return Output(the_output[:len(self.outputs)],keys=self.compute_output_order[:len(self.outputs)])
    def add_target(self,variable,error_func=lambda x,y: T.mean((x-y)**2),name='target',bcast=(True,True,True)):
        tp = T.TensorType(variable.type.dtype, variable.type.broadcastable)
        v = as_input(tp(name),name=name)
        er = error_func(variable,v)
        er.name='output'
        error_node = N(er,model=self,name='ErrorNode')
        e = self.outputs.append(error_node.output)
        variable.__dict__['error_functions'] = variable.__dict__.get('error_functions',[])
        variable.__dict__['error_functions'].append(e)
        return v,er
    def add_update(self,variable,error_term,opt_func = lambda x,y: x+as_parameter(theano.shared(0.001),name='learning_rate',initialized=True)*y):
        variable.__dict__['updates'] = variable.__dict__.get('error_functions',[])
        g = T.grad(error_term,variable)
        g.name = 'gradient'
        variable.__dict__['updates'].append(opt_func(variable,g))
    def add_gradient_descent(self,v_to_change,v_to_target=None):
        if v_to_target is None:
            if len(self.outputs) > 1:
                raise Exception('Target variable is not provided and the model has no outputs.')
            v_to_target = self.output
        v,er = self.add_target(v_to_target)
        self.add_update(v_to_change,er)
        return v

class Runner(object):
    """
        A runner object holds the theano compiled function and the associated input/output mappings.

        Also a state can be either per model or per Runner.

        To decide: does a runner have a copy of all variables or references?
    """
    def __init__(self,model):
        self.model=model
        self.input_order = [] #f7(additional_inputs + filter(is_input,variables) + filter(is_input_parameter,variables))
        self.additional_inputs = [] #additional_inputs
        self.state_inits = [] #?
        self.output_order = [] # f7(self.outputs + [v.state_out_state for v in state_variables])
        self.updates = theano.updates.OrderedUpdates()
        self.givens = [] # [(a,b) for (a,b) in givens.items()]
        self.debug = False
    def create(self):
        if self.debug:
            print 'solving for:',self.outputs
            print 'all variables:',len(variables)
            print 'input:',filter(is_input,variables)
            print 'parameters:',filter(is_input_parameter,variables)
            print 'states:',state_variables
            print 'updates:',self.compute_updates_order
        self.input_dict = dict((v,None) for v in self.input_order)
        self.state_dict = dict((v,None) for v in self.output_order if is_out_state(v))
        self.compute = theano.function(inputs=self.input_order, 
                                        outputs=self.output_order, 
                                        updates=self.updates,
                                        givens=self.givens,
                                        on_unused_input='ignore')
    def run(self,inp):
        ## TODO: adapt this to the context
        c = O()
        c.input = the_input
        c.model = self
        #c.config = self.config
        input_dict = {}
        input_dict.update(dict(zip(self.additional_inputs,additional_inputs)))
        for k,v in self.compute_input_dict.items():
            if v is not None:
                if hasattr(v,'array'):
                    input_dict[k] = v
                elif hasattr(v,'get'):
                    input_dict[k] = v.get(len(the_input))
            else:
                if is_input(k):
                    if k not in self.additional_inputs:
                        input_dict[k] = the_input
                if is_input_parameter(k):
                    input_dict[k] = k.param_init(create_context_O(k,model=self,input=the_input))
                if is_state(k):
                    if self.compute_state_dict.get(k.state_out_state,None) is None:
                        input_dict[k] = k.state_init(create_context_O(k,model=self,input=the_input))
                    else:
                        input_dict[k] = self.compute_state_dict[k.state_out_state]
        for shared_parameter in self.parameters._all:
            if not hasattr(shared_parameter,'initialized') or shared_parameter.initialized == False:
                shared_parameter.set_value(shared_parameter.param_init(create_context_O(shared_parameter,model=self,input=the_input)))
                shared_parameter.initialized = True
        the_vars = [input_dict[v] for v in self.compute_input_order]
        the_output = self.compute(*the_vars)
        for (o,k) in zip(the_output, self.compute_output_order):
            if is_out_state(k):
                self.compute_state_dict[k] = o
        return the_output[:len(self.outputs)]


