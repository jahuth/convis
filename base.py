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

def f7(seq):
    """ This function is removing duplicates from a list while keeping the order """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


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
        if has_convis_attribute(self.graph,'root_of'):
            return get_convis_attribute(self.graph,'root_of')
        if not hasattr(self.graph,'__is_convis_var'):
            self.graph = as_output(T.as_tensor_variable(self.graph))
        if self.graph.name is None:
            # we only replace the name if it is necessary
            self.graph.name = 'output'
        set_convis_attribute(self.graph,'root_of', self)
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
            if not get_convis_attribute(v,'path',None) is None:
                if get_convis_attribute(v,'path')[0] == self:
                    continue # we already are the owner of this variable
            if has_convis_attribute(v,'full_name'):
                set_convis_attribute(v,'full_name', self.name+'.'+get_convis_attribute(v,'full_name',''))
                set_convis_attribute(v,'path', [self] + get_convis_attribute(v,'path',[v]))
            else:
                set_convis_attribute(v,'full_name', self.name+'.'+get_convis_attribute(v,'name',''))
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
                set_convis_attribute(v,'simple_name', v.name)
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
        for o in [get_convis_attribute(v,'state_out_state') for v in filter(is_state,self.get_variables())]:
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
            if replace_inputs and has_convis_attribute(input.owner.inputs[0].owner.inputs[1].owner.inputs[0],'replaceable_input'):
                input.owner.inputs[0].owner.inputs[1] = v.dimshuffle(('x',0,1,2))
            else:
                input.owner.inputs[0].owner.inputs.append(v.dimshuffle(('x',0,1,2)))
            if do_debug:
                print 'inputs are now:',input.owner.inputs[0].owner.inputs
            if get_convis_attribute(v, 'connects', None) is None:
                set_convis_attribute(v, 'connects',[])
            get_convis_attribute(v,'connects').append([self,other])
        else:
            if has_convis_attribute(input,'variable_type') and get_convis_attribute(input,'variable_type') == 'input':
                set_convis_attribute(input,'variable_type','replaced_input')
            if get_convis_attribute(v, 'connects',None) is None:
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

class _Vars(O):
    def __init__(self,model,**kwargs):
        self._model = model
        super(_Vars,self).__init__(**kwargs)
        vars = [v for v in self._model.get_variables()  if v.name is not None]
        nodes = f7([v.node for v in vars if hasattr(v,'node')])
        for n in nodes:
            self.__dict__[save_name(n.name)] = O(**dict([(save_name(k.simple_name),k) for k in vars if has_convis_attribute(k,'node') and get_convis_attribute(k,'node') == n]))
        self.__dict__['_all'] = O(**dict([(full_path(k),k) for k in vars if has_convis_attribute(k,'path')]))
        self.__dict__['_search'] = _Search(**dict([(full_path(k),k) for k in vars if has_convis_attribute(k,'path')]))

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
        self.resolution = ResolutionInfo(pixel_per_degree,steps_per_second)
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
        return self.resolution.degree_to_pixel(degree)
    def pixel_to_degree(self,pixel):
        return self.resolution.pixel_to_degree(pixel)
    def seconds_to_steps(self,t):
        return self.resolution.seconds_to_steps(t)
    def steps_to_seconds(self,steps):
        return self.resolution.steps_to_seconds(steps)
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
        if has_convis_attribute(a,'node'):
            self.add(get_convis_attribute(a,'node'))
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
    def create_function(self,updates=None,additional_inputs=[]):
        if self.debug is not True:
            # we disable warnings from theano gof because of the unresolved cache leak
            # TODO: fix the leak and re-enable warnings
            import logging
            logging.getLogger("theano.gof.cmodule").setLevel(logging.ERROR) 
            ##pass
        if updates is None:
            updates = theano.updates.OrderedUpdates()
        for a,b in  self.mappings.items():
            for n in self.nodes:
                if get_convis_attribute(v,'variable_type') == 'input':
                    set_convis_attribute(v,'variable_type', 'replaced_input')
                theano_utils.replace(n.graph,a,b)
        outputs = [o._as_TensorVariable() if hasattr(o,'_as_TensorVariable') else o for o in self.outputs]
        variables = f7([v for o in outputs for v in theano_utils.get_variables_iter(o)])
        for v in variables:
            if has_convis_attribute(v,'updates'):
                if v in updates:
                    updates[v] = T.sum([updates[v]]+[u for u in get_convis_attribute(v,'updates')])
                else:
                    updates[v] = T.sum([u for u in get_convis_attribute(v,'updates')])
        self.compute_input_order = f7(additional_inputs + filter(is_input,variables) + filter(is_input_parameter,variables))
        self.additional_inputs = additional_inputs
        self.compute_state_inits = []
        state_variables = filter(is_state,variables)
        self.compute_output_order = f7(outputs + [get_convis_attribute(v,'state_out_state') for v in state_variables])
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
                    input_dict[k] = get_convis_attribute(k,'param_init')(create_context_O(k,input=the_input))
                if is_state(k):
                    if self.compute_state_dict.get(get_convis_attribute(k,'state_out_state'),None) is None:
                        input_dict[k] = get_convis_attribute(k,'state_init')(create_context_O(k,input=the_input))
                    else:
                        input_dict[k] = self.compute_state_dict[get_convis_attribute(k,'state_out_state')]
        for shared_parameter in self.parameters._all:
            if is_shared_parameter(shared_parameter):
                #if (not hasattr(shared_parameter,'initialized') or shared_parameter.initialized == False) and not hasattr(shared_parameter,'optimized'):
                # until we can track which config values where changed, we re-initialize everything
                # all smart stuff is now in the injected .update method
                get_convis_attribute(shared_parameter,'update')(create_context_O(shared_parameter,input=the_input))
                #shared_parameter.initialized = True
        the_vars = [input_dict[v] for v in self.compute_input_order]
        the_output = self.compute(*the_vars)
        if run_after is not None:
            run_after(the_output,self)
        for (o,k) in zip(the_output, self.compute_output_order):
            if is_out_state(k):
                self.compute_state_dict[k] = o
        return Output(the_output[:len(self.outputs)],keys=self.compute_output_order[:len(self.outputs)])

    # methods for adding optimization methods easily
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


