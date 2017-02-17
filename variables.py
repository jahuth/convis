import theano
import theano.tensor as T
import new
from debug import *
from o import O

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
