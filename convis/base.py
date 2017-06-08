from misc_utils import unique_list, suppress

from .imports import theano
from .imports import T
import numpy as np
import matplotlib.pylab as plt
from .theano_utils import conv3d, conv2d
import uuid
from . import retina_base
from . import io
from . import theano_utils
from exceptions import NotImplementedError
from variable_describe import describe, describe_dict, describe_html
import warnings

import debug
reload(debug)
from debug import *

import variables
reload(variables)
from variables import *
import o
from o import O, Ox, save_name
from collections import OrderedDict

### Node and Model classes

def len_parents(n):
    if hasattr(n,'parent') and n.parent != n:
        return len_parents(n.parent)+1
    return 0

class GraphWrapper(object):
    """
        `GraphWrapper` wraps a theano graph and provides
        labeling to the named variables within.

        The graph within the `GraphWrapper` is accessible through
        the `.graph` attribute.
        
        If the graph has no inputs (only shared variables),
        `.compute()` compiles the graph and gives its value.

        `.add_input(var,input=i)` can be used to add `var` to an input
        variable `i`. This will either add `var` to the input `i` if it is
        a sum, or replace it with `var`. `i` can be the input variable 
        itself or its name or `None`. If `input=None`, the `default_input`
        attribute of the `GraphWrapper` will be used.

        A `GraphWrapper` (and also `N` layer objects) will try to behave 
        similar to a theano variable in many contexts.
        Mathematical operations, such as `+`, `-`, etc. create a new theano
        variable, the same as if it was applied to the graph within the
        `GraphWrapper`.

    """
    parent = None
    config_dict = {}
    node_type = 'Node'
    node_description = ''
    expects_config = False
    def __init__(self,graph,name,m=None,parent=None,ignore=[],scan_op=None,**kwargs):
        self.m = m
        self.parent = parent
        self.graph = graph
        #if has_convis_attribute(self.graph,'root_of'):
        #    return get_convis_attribute(self.graph,'root_of')
        if not is_var(self.graph):
            self.graph = as_output(T.as_tensor_variable(self.graph))
        if get_convis_attribute(self.graph,'name') is None:
            # we only replace the name if it is necessary
            set_convis_attribute(self.graph,'name','output')
        set_convis_attribute(self.graph,'root_of', self)
        self.name = name
        self.ignore = ignore
        self.scan_op = scan_op
        self.__dict__.update(kwargs)
        self.follow_scan = True
        if self.follow_scan and scan_op is None:
            self.wrap_scans(self.graph)
        self.label_variables(self.graph)
    def get_model(self):
        if hasattr(self,'model'):
            return self.model
        if self.parent is not None:
            return self.parent.get_model()
    def get_config(self):
        return self.config_dict
    def get_config_value(self,key=None,default=None,type_cast=None):
        if type_cast is not None:
            return type_cast(self.get_config_value(key,default))
        return self.config.get(key,default)
    def set_config(self,key):
        if hasattr(key,'get'):
            send_dbg('set_config',str(getattr(self,'name',''))+' replaced config: '+str(key)+'',1)
            self.config_dict = dict(key)
        elif type(key) is str:
            self.name = key
            self.config_dict['name'] = key
        else:
            raise Exception('Config supplied is not a dict and not a name!')
    def set_config_value(self,key,v='__set_name__'):
        send_dbg('set_config',str(getattr(self,'name',''))+' set config: '+str(key)+': '+str(v),1)
        self.config_dict[key] = v
    def compute(self):
        self.f = theano.function([],self.graph)
        return self.f()
    @property
    def config(self):
        if self.config_dict is None:
            if self.parent is not None:
                return self.parent.config
            if self.expects_config:
                raise Exception('GraphWrapper '+str(getattr(self,'name','??'))+' has no configuration! But also no parent!')
            return {}
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
        for i,ow in enumerate(unique_list([v.owner for v in my_scan_vars])):
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
            if get_convis_attribute(v,'name') is None:
                set_convis_attribute(v,'name','scan_output_'+str(i))
            as_output(v)
        for v in my_named_vars + [v for v in my_scan_vars if v.owner.op != self.scan_op]:
            if not get_convis_attribute(v,'path',None) is None:
                if get_convis_attribute(v,'path')[0] == self:
                    continue # we already are the owner of this variable
            if has_convis_attribute(v,'full_name'):
                set_convis_attribute(v,'full_name', self.name+'.'+get_convis_attribute(v,'full_name',''))
                set_convis_attribute(v,'path', [self] + get_convis_attribute(v,'path',[v]))
            else:
                set_convis_attribute(v,'full_name', self.name+'.'+str(get_convis_attribute(v,'name','')))
                set_convis_attribute(v,'path', [self, v])
            global do_debug
            if do_debug:
                print 'labeled: ',get_convis_attribute(v,'path')
            if get_convis_attribute(v,'node',None) != None and get_convis_attribute(v,'node') != self:
                if get_convis_attribute(v,'node').parent is None:
                    get_convis_attribute(v,'node').parent = self
            else:
                set_convis_attribute(v,'node', self)
            if not get_convis_attribute(v,'simple_name', None) is None:
                set_convis_attribute(v,'simple_name', get_convis_attribute(v,'name'))
            #v.node = self
    @property
    def shape(self):
        return self.graph.shape
    @property
    def reshape(self):
        return self.graph.reshape
    def _as_TensorVariable(self):
        """
            When theano uses an object as a variable it will first check if it supports this function.
            We simply act as if we are the annotated graph that is stored inside.

            Note from theano source: # TODO: pass name and ndim arguments
        """
        return self.graph
    def filter_variables(self,filter_func=is_input):
        return create_hierarchical_Ox(filter(filter_func,theano_utils.get_named_variables_iter(self.graph)),pi=len_parents(self))
    @property
    def output(self):
        return self.graph
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
    def save_parameters_to_json(self,filename):
        io.save_dict_to_json(filename,dict(self.parameters._all.__iteritems__()))
    def load_parameters_from_json(self,filename,strict=True):
        dat = io.load_dict_from_json(filename)
        if strict:
            assert(set(dat.keys()) == set(self.parameters._all.__iterkeys__()), 'Entries do not match. Are you sure that the parameters are from the same subgraph?')
        for p, param in self.parameters._all.__iteritems__():
            if p in dat.keys():
                param.set_value(dat[p])
            else:
                raise Exception('Value not found in saved parameters! Are you sure that the parameters are from the same subgraph?')
    def __repr__(self):
        if hasattr(self, 'node_description') and callable(self.node_description):
            # we can provide a dynamic description
            return '['+str(self.node_type)+'] ' + getattr(self,'name','??') + ': ' + str(self.node_description())
        return '['+str(self.node_type)+'] ' + getattr(self,'name','??') + ': ' + str(self.node_description)
    def replace(self,a,b):
        for o in self.graph:
            theano_utils._replace(o,b,a)
        # replace in out states
        for o in [get_convis_attribute(v,'state_out_state') for v in filter(is_state,self.get_variables())]:
            theano_utils._replace(o,b,a)
    def shared_parameter(self, f=lambda x:x, name='',**kwargs):
        # todo: where to save config?
        if 'config_key' in kwargs.keys() and 'config_default' in kwargs.keys():
            if not callable(f):
                raise Exception('Need to implement this!')
            return shared_parameter(f,
                                    O()(node=self,
                                        model=getattr(self,'model',None),
                                        resolution= self.model.resolution if hasattr(self,'model') else variables.default_resolution,
                                        get_config=self.get_config,
                                        get_config_value=self.get_config_value,
                                        value_from_config=lambda: self.get_config_value(kwargs.get('config_key'),kwargs.get('config_default')),
                                        value_to_config=lambda v: self.set_config_value(kwargs.get('config_key'),v)),
                                    name=name,**kwargs)
        return shared_parameter(f,O()(node=self,model=getattr(self,'model',None),get_config=self.get_config,get_config_value=self.get_config_value),name=name,**kwargs)
    def add_input(self,other,replace_inputs=do_replace_inputs,input=None):
        if input is None:
            if hasattr(self, 'default_input'):
                input = self.default_input
            elif hasattr(self.variables, 'input'):
                input = self.variables.input
            else:
                raise Exception('No input found in '+getattr(self,'name','[unnamed node]')+'!')
        if type(input) is str:
            """if we have more than one input, we have to specify which one we want to add something to"""
            if input in self.inputs.keys():
                input = self.inputs[input]
            else:
                raise Exception('Input "'+str(input)+'"" found in '+getattr(self,'name','[unnamed node]')+' inputs!')
        if do_debug:
            print 'connecting',other.name,'to',getattr(self,'name','??'),''
        v = other
        if hasattr(other,'_as_TensorVariable'):
            v = other._as_TensorVariable()
        if theano_utils.add_input_to_a_sum_op(input, v, replace_inputs=replace_inputs):
            if get_convis_attribute(v, 'connects', None) is None:
                    set_convis_attribute(v, 'connects',[])
            get_convis_attribute(v,'connects').append([self,other])
        else:
            # fallback to replacing one variable with another
            #  (this is deprecated behaviour)
            if has_convis_attribute(input,'variable_type') and get_convis_attribute(input,'variable_type') == 'input':
                set_convis_attribute(input,'variable_type','replaced_input')
            if get_convis_attribute(v, 'connects', None) is None:
                    set_convis_attribute(v, 'connects',[])
            get_convis_attribute(v,'connects').append([self,other])
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
    def debugprint(self):
        theano.printing.debugprint(self.graph)

class Layer(GraphWrapper):
    """
        The `Layer` class
        ------------------

        A Layer is a wrapper around a theano graph that has to contain
        a input (3d or 5d) and keeps its configuration.

        A simpler wrapper is `GraphWrapper`.

        Minimal Usage::

            class A_New_Layer(N):
                def __init__(self,config={},name=None,model=None):
                    self.model = model
                    self.set_config(config)
                    my_input = self.create_input()
                    super(ANewLayer,self).__init__(my_input,name=name)

        

    """
    states = {}
    state_initializers = {}
    inputs = OrderedDict()
    expects_config = False
    def __init__(self,graph,name=None,m=None,parent=None,config=None,inputs=None,**kwargs):
        self.m = m
        self.parent = parent
        self.node_type = 'Node'
        self.node_description = ''
        if config is not None:
            self.set_config(config)
        elif self.expects_config:
            raise Exception('No config for node '+str(getattr(self,'name','??'))+'! Use .set_config(\{\}) before calling super constructor!')
        if name is None:
            if self.get_config_value('name', None) is None:
                name = str(uuid.uuid4()).replace('-','')
                self.set_config_value('name', name)
            else:
                name = self.get_config_value('name')
        if not hasattr(self,'default_input'):
            raise Exception('No input defined for node '+str(getattr(self,'name','??'))+'! Use .create_input(...) before calling super constructor!')
        super(N, self).__init__(graph,name=name,m=m,parent=parent)
        if inputs is not None:
            if type(inputs) is list:
                for inp in inputs:
                    self.add_input(inp)
            else:
                self.add_input(inputs)
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
    def create_input_5d(self,n=1,name='input',sep='_'):
        if n == 1:
            self.input = T.sum([as_input(theano_utils.dtensor5(),name,replaceable_input=True)],axis=1)
            self.default_input = self.input
            self.inputs.update(OrderedDict([(name,self.input)]))
            return self.input
        elif type(n) == int:
            self.inputs.update(OrderedDict([(name+sep+str(i),T.sum([as_input(theano_utils.dtensor5(),name+sep+str(i),replaceable_input=True)],axis=1)) for i in range(n)]))
            self.default_input = self.inputs.values()[0]
            self.input = T.join(*([0]+self.inputs.values()))
            return self.inputs.values()
        elif type(n) in [list,tuple]:
            self.inputs.update(OrderedDict([(input_name,T.sum([as_input(theano_utils.dtensor5(),str(input_name),replaceable_input=True)],axis=1)) for input_name in n]))
            self.default_input = self.inputs[n[0]]
            self.input = self.inputs[n[0]]
            return self.inputs
        else:
            raise Exception('Argument not understood. Options are: an int (either 1 for a single input or >1 for more) or a list of names for the inputs.')
    def get_config_value(self,key,default=None,type_cast=None):
        if type_cast is not None:
            return type_cast(self.get_config_value(key,default))
        if not hasattr(self,'config'):
            return default
        return self.config.get(key,default)
    def shared_parameter(self, f=lambda x:x, name='',**kwargs):
        if 'config_key' in kwargs.keys() and 'config_default' in kwargs.keys():
            return shared_parameter(f,
                                    O()(node=self,
                                        model=self.model,
                                        resolution=self.model.resolution,
                                        get_config=self.get_config,
                                        get_config_value=self.get_config_value,
                                        value_from_config=lambda: self.get_config_value(kwargs.get('config_key'),kwargs.get('config_default')),
                                        value_to_config=lambda v: self.set_config_value(kwargs.get('config_key'),v)),
                                    name=name,**kwargs)
        return shared_parameter(f,O()(node=self,model=self.model,get_config=self.get_config,get_config_value=self.get_config_value),name=name,**kwargs)
    def shape(self,input_shape):
        # unless this node does something special, the shape of the output should be identical to the input
        return input_shape
N = Layer

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
            self._out_dict_by_short_names = OrderedDict([(save_name(get_convis_attribute(k,'name')),o) for (k,o) in zip(keys,outs) if has_convis_attribute(k,'name') and type(get_convis_attribute(k,'name')) is str])
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


class Model(object):
    """
        M: The model class
        ------------------

        Creates a model that can compute a set of outputs.

        Minimal Usage::

            m = Model()
            m.add_output(some_theano_variable_or_convis_layer)
            some_output = m.run(some_input)

        When calling `run` the first time, the graph attached to 
        `some_theano_variable_or_convis_layer` will be compiled.
        To recompile the function or compile it with a certain order
        of parameters use `create_function`.

        Long input should be supplied to `run_in_chunks` instead of `run`.

    """
    def __init__(self, size=(10,10), pixel_per_degree=10.0, steps_per_second= 1000.0, filter_epsilon=0.01, **kwargs):
        self.debug = False
        self.mappings = {}
        self.givens = {}
        self.outputs = []
        self.config = {}
        self.resolution = ResolutionInfo(pixel_per_degree=pixel_per_degree,steps_per_second=steps_per_second,filter_epsilon=filter_epsilon)
        self.size_in_degree = size
        self.__dict__.update(kwargs)
    @property
    def parameters(self):
        return create_hierarchical_Ox(filter(is_shared_parameter,theano_utils.get_named_variables_iter(self.outputs)),pi=len_parents(self))
    @property
    def states(self):
        return create_hierarchical_Ox(filter(is_state,theano_utils.get_named_variables_iter(self.outputs)),pi=len_parents(self))
    @property
    def variables(self):
        return create_hierarchical_Ox(theano_utils.get_named_variables_iter(self.outputs),pi=len_parents(self))
    def save_parameters_to_json(self,filename):
        io.save_dict_to_json(filename,dict(self.parameters._all.__iteritems__()))
    def load_parameters_from_json(self,filename,strict=True):
        dat = io.load_dict_from_json(filename)
        if strict:
            assert(set(dat.keys()) == set(self.parameters._all.__iterkeys__()), 'Entries do not match. Are you sure that the parameters are from the same model?')
        for p, param in self.parameters._all.__iteritems__():
            if p in dat.keys():
                param.set_value(dat[p])
            else:
                raise Exception('Value not found in saved parameters! Are you sure that the parameters are from the same model?')
    def degree_to_pixel(self,degree):
        return self.resolution.degree_to_pixel(degree)
    def pixel_to_degree(self,pixel):
        return self.resolution.pixel_to_degree(pixel)
    def seconds_to_steps(self,t):
        return self.resolution.seconds_to_steps(t)
    def steps_to_seconds(self,steps):
        return self.resolution.steps_to_seconds(steps)
    def add_output(self,a,name=None):
        if hasattr(a,'graph'):
            a = a.graph
        if getattr(a,'name',None) is None and name is not None:
            a = as_output(a)
            set_convis_attribute(a, 'name', name)
        self.outputs.append(a)
    def in_out(self,a,b):
        #self.map(b.var('input'),a.var('output'))
        if hasattr(a,'graph'):
            a = a.graph
        print 'Replacing:',a,b
        if issubclass(b.__class__, N):
            if has_convis_attribute(b.var('input'),'variable_type') and get_convis_attribute(b.var('input'),'variable_type') == 'input':
                set_convis_attribute(b.var('input'),'variable_type','replaced_input')
            #theano_utils._replace(b.output,b.var('input'),a.var('output'))
            if not has_convis_attribute(a, 'connects'):
                set_convis_attribute(a,'connects',[])
            get_convis_attribute(a,'connects').append([b,get_convis_attribute(a,'node',a)])
            try:
                theano_utils._replace(b.output,b.variables.input,a)
            except:
                print 'Something not found! ',a,b.variables
        elif has_convis_attribute(b,'node'):
            if not has_convis_attribute(a, 'connects'):
                set_convis_attribute(a,'connects',[])
            get_convis_attribute(a,'connects').append([get_convis_attribute(b,'node'),get_convis_attribute(a,'node',a)])
            theano_utils._replace(get_convis_attribute(b,'node').output,b,a)
        else:
            raise Exception('This is not a node and not a variable with a node. Maybe the variable was not named?')
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
                if get_convis_attribute(v,'variable_type') == 'input':
                    set_convis_attribute(v,'variable_type', 'replaced_input')
                theano_utils.replace(n.graph,a,b)
        outputs = [o._as_TensorVariable() if hasattr(o,'_as_TensorVariable') else o for o in self.outputs]
        variables = unique_list([v for o in outputs for v in theano_utils.get_variables_iter(o)])
        for v in variables:
            if has_convis_attribute(v,'updates'):
                if v in updates:
                    updates[v] = T.sum([updates[v]]+[u for u in get_convis_attribute(v,'updates')],axis=0)
                else:
                    updates[v] = T.sum([u for u in get_convis_attribute(v,'updates')],axis=0)
        self.compute_input_order = unique_list(additional_inputs + filter(is_input,variables) + filter(is_input_parameter,variables))
        self.additional_inputs = additional_inputs
        self.compute_state_inits = []
        state_variables = filter(is_state,variables)
        self.compute_output_order = unique_list(outputs + [get_convis_attribute(v,'state_out_state') for v in state_variables])
        self.compute_updates_order = theano.updates.OrderedUpdates()
        self.compute_updates_order.update(updates)
        for state_var in state_variables:
            self.compute_input_order.append(state_var)
            #self.compute_updates_order[state_var] = state_var.state_out_state
            #print state_var.state_out_state
            self.compute_state_inits.append(get_convis_attribute(state_var,'state_init'))
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
            if is_shared_parameter(shared_parameter) and has_convis_attribute(shared_parameter,'initialized'):
                set_convis_attribute(shared_parameter, 'initialized', False)
    def run_in_chunks(self,the_input,max_length,additional_inputs=[],inputs={},run_after=None,**kwargs):
        """
            To run a simulation that is longer than your memory is large,
            the input has to be chunked into smaller pieces.

            This function computes chunks of `max_length` at a time and 
            returns the concatenated output (scalar outputs will be a vector 
            of return values, one for each chunk).

            `additional_inputs`, `inputs` and `run_after` will be forwarded
            to the run function.

                `inputs` is a dictionary mapping input variables to their values

                `additional_inputs` is a list of input values that has to match
                the order of `additional_inputs` supplied to `create_function`.
                If you did not use `additional_inputs` when you created the function
                or you do not remember the order, use the `input` dictionary
                instead!

                `run_after` is None or a 2 argument python function that will 
                be executed after each chunk. It will recieve the output as 
                a first argument and the model as a second argument.

            All inputs will be truncated according  to the current chunk, so the
            complete time series has to be supplied for all inputs.

        """
        chunked_output = [] # this will store a list of lists of our chunked results
        t = 0
        while t < the_input.shape[0]:
            if self.debug:
                print 'Chunk:', t, t+max_length
            oo = self.run(the_input[t:(t+max_length)],
                          additional_inputs=[ai[t:(t+max_length)] for ai in additional_inputs],
                          inputs=dict([(k,i[t:(t+max_length)]) for k,i in inputs.items()]),
                          run_after=run_after)
            for i,o in enumerate(oo):
                while len(chunked_output) < i+1:
                    chunked_output.append([])
                chunked_output[i].append(o)
            t += max_length
        outs = []
        for co in chunked_output:
            try:
                outs.append(np.concatenate(co,axis=0))
            except:
                outs.append(np.array(co))
        return Output(outs,keys=self.compute_output_order[:len(self.outputs)])
    def run_in_chuncks(self,the_input,max_length,additional_inputs=[],inputs={},run_after=None,**kwargs):
        """typo"""
        return self.run_in_chunks(the_input,max_length,additional_inputs=additional_inputs,inputs=inputs,run_after=run_after,**kwargs)
    def run(self,the_input,additional_inputs=[],inputs={},run_after=None,**kwargs):
        """
            Runs the model

                `the_input` is the main input to the model. All inputs that were not
                set explicitly, will recieve this value!

                `inputs` is a dictionary mapping input variables to their values

                `additional_inputs` is a list of input values that has to match
                the order of `additional_inputs` supplied to `create_function`.
                If you did not use `additional_inputs` when you created the function
                or you do not remember the order, use the `input` dictionary
                instead!

                `run_after` is None or a 2 argument python function that will 
                be executed after the computation finishes. It will recieve the output as 
                a first argument and the model as a second argument.
        """
        if not hasattr(self,'compute_input_dict'):
            # todo: manage multiple functions
            self.create_function()
        if self.debug:
            print "Using supplied inputs: ", inputs.keys()
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
                    input_dict[k] = get_convis_attribute(k,'param_init')(create_context_O(k,input=the_input,model=self,resolution=self.resolution))
                if is_state(k):
                    if self.compute_state_dict.get(get_convis_attribute(k,'state_out_state'),None) is None:
                        input_dict[k] = get_convis_attribute(k,'state_init')(create_context_O(k,input=the_input,model=self,resolution=self.resolution))
                    else:
                        input_dict[k] = self.compute_state_dict[get_convis_attribute(k,'state_out_state')]
        for shared_parameter in self.parameters._all:
            if is_shared_parameter(shared_parameter):
                #if (not hasattr(shared_parameter,'initialized') or shared_parameter.initialized == False) and not hasattr(shared_parameter,'optimized'):
                # until we can track which config values were changed, we re-initialize everything
                # all smart stuff is now in the injected .update method
                if (get_convis_attribute(shared_parameter,'initialized',False) is False) and not has_convis_attribute(shared_parameter,'optimized'):
                    get_convis_attribute(shared_parameter,'update')(create_context_O(shared_parameter,input=the_input,model=self,resolution=self.resolution))
                #shared_parameter.initialized = True
        the_vars = [input_dict[v] for v in self.compute_input_order]
        the_output = self.compute(*the_vars)
        if run_after is not None:
            run_after(the_output,self)
        for (o,k) in zip(the_output, self.compute_output_order):
            if is_out_state(k):
                self.compute_state_dict[k] = o
        return Output(the_output[:len(self.outputs)],keys=self.compute_output_order[:len(self.outputs)])

    def draw_simple_diagram(self, connector = ' -> '):
        """
            Prints a very simple horizontal tree diagram of the connected `N` nodes.

            Note: This diagram only shows connections that were created between 
            `N` nodes by `add_input` (or `+=`).
            
            Example:: 

                print draw_simple_diagram(retina, connector=' - ')

            Output::

                input - OPL - Bipolar - GanglionInputLayer_Parvocellular_On - GanglionSpikes__Parvocellular_On - output
                                      - GanglionInputLayer_Parvocellular_Off - GanglionSpikes__Parvocellular_Off - output
        """
        node_dict = {}
        for v in filter(lambda x: has_convis_attribute(x,'connects'), self.variables._all):
            for c in get_convis_attribute(v,'connects'):
                node_dict.setdefault(c[1],[]).append(c[0])
        possible_start_nodes = node_dict.keys()
        possible_end_nodes = reduce(lambda x,y: x+y, node_dict.values())
        true_start_nodes = []
        for e in possible_start_nodes:
            if not e in possible_end_nodes:
                true_start_nodes.append(e)
        def simple_model_diagram(n,offset=0,prev_offset=0):
            if type(n) is list:
                return '\n'.join(['input'+connector+simple_model_diagram(i,offset=offset,prev_offset=prev_offset+len('input'+connector)) for i in n])
            s = save_name(getattr(n,'name','Layer'))
            if n not in node_dict:
                s += connector+'output\n'
                return s
            next_offset = prev_offset+len(save_name(n.name))
            for i,child in enumerate(node_dict[n]):
                if i > 0:
                    s += ' '*next_offset
                s+= connector
                s+= simple_model_diagram(child,offset=prev_offset,prev_offset=next_offset+len(connector))
            return s
        diagram = simple_model_diagram(true_start_nodes)
        return diagram
    def debugprint(self):
        theano.printing.debugprint(self.outputs)
    # methods for adding optimization methods easily
    def add_target(self,variable,error_func= None,name='target',bcast=(True,True,True)):
        """
            Default error func: lambda x,y: T.mean((x-y)**2)
        """
        if error_func is None:
            error_func = lambda x,y: T.mean((x-y)**2)
        tp = T.TensorType(variable.type.dtype, variable.type.broadcastable)
        v = as_input(tp(name),name=name)
        er = error_func(variable,v)
        set_convis_attribute(er,'name','output')
        error_node = GraphWrapper(er,config={},model=self,name='ErrorNode',ignore=[variable])
        e = self.outputs.append(error_node.output)
        variable.__dict__['error_functions'] = variable.__dict__.get('error_functions',[])
        variable.__dict__['error_functions'].append(e)
        return v,er
    def add_update(self,variable,error_term,opt_func = None):
        """
            Default opt_func: lambda x,y: x-as_parameter(theano.shared(0.001),name='learning_rate',initialized=True)*y
        """
        if opt_func is None:
            opt_func = lambda x,y: x-as_parameter(theano.shared(0.001),name='learning_rate',initialized=True)*y
        variable.__dict__['updates'] = variable.__dict__.get('error_functions',[])
        g = T.grad(error_term,variable)
        set_convis_attribute(g,'name','gradient')
        self.outputs.append(opt_func(variable,g)) # instead we should also explore all updates :/
        # but that means we have to explore all variables after we explored all varaibles etc.
        variable.__dict__['updates'].append(opt_func(variable,g))
    def add_gradient_descent(self,v_to_change,v_to_target=None,error_func=None,opt_func = None):
        if v_to_target is None:
            if len(self.outputs) > 1:
                raise Exception('Target variable is not provided and the model has no outputs.')
            v_to_target = self.graph
        v,er = self.add_target(v_to_target,error_func=error_func)
        self.add_update(v_to_change,er,opt_func=opt_func)
        return v
    def _repr_html_(self):
        s = ""
        s += "<h1>"+str(getattr(self,'name','(unnamed model)'))+"</h1>"
        s += "Diagram of this model:<br><pre>"+str(self.draw_simple_diagram())+"</pre>"
        return s

M = Model

def make_model(outputs,name='Model',**kwargs):
    if type(outputs) not in [list,tuple]:
        outputs = [outputs]
    m = Model(name=name,**kwargs)
    for o in outputs:
        m.add_output(o)
    m.create_function()
    return m
