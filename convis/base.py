from __future__ import print_function
from .misc_utils import unique_list, suppress

import numpy as np
import matplotlib.pylab as plt
import uuid
from . import retina_base
from . import io
try:
    from exceptions import NotImplementedError
except ImportError:
    pass
from .variable_describe import describe, describe_dict, describe_html
import warnings

from . import debug
from .debug import *

from . import variables
from .variables import *
from . import o
from .o import O, Ox, save_name
from collections import OrderedDict

# ----

import torch
import numpy as np
from torch.autograd import grad
from torch import nn

try:
    from functools import reduce
except:
    pass

from .variables import Variable, State, Parameter, as_parameter, is_variable

### Node and Model classes

def len_parents(n):
    if hasattr(n,'parent') and n.parent != n:
        return len_parents(n.parent)+1
    return 0

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
        
class Layer(torch.nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self._variables = []
        self._use_cuda = False
    def cuda(self):
        self._use_cuda = True
        super(Layer, self).cuda()
    def cpu(self):
        self._use_cuda = False
        super(Layer, self).cpu()
    def __call__(self,*args,**kwargs):
        new_args = []
        for a in args:
            if not hasattr(a, 'data'):
                if not hasattr(a, 'numpy'):
                    a = torch.autograd.Variable(a)
                else:
                    a = torch.autograd.Variable(torch.Tensor(a))
            if hasattr(self,'dims'):
                if self.dims == 5:
                    if len(a.data.shape) == 3:
                        a = a[None,None,:,:,:]
                if self.dims == 3:
                    if len(a.data.shape) == 5:
                        a = a[0,0,:,:,:]
            new_args.append(a)
        o = super(Layer, self).__init__(*new_args,**kwargs)
        return o
    @property
    def params(self):
        # see https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
        # for full example on how to find parameters
        return Ox(self.named_parameters())
    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for var in self._variables:
            if var is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                var.data = fn(var.data)
                if var._grad is not None:
                    var._grad.data = fn(var._grad.data)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self
    def __setattr__(self, name, value):
        if is_variable(value):
            self._variables.append(value)
            self.__dict__[name] = value
        else:
            super(Layer, self).__setattr__(name, value)

class Model(object):
    """

    """
    def __init__(self, pixel_per_degree=10.0, steps_per_second= 1000.0, filter_epsilon=0.01, **kwargs):
        self.debug = False
        self.resolution = ResolutionInfo(pixel_per_degree=pixel_per_degree,steps_per_second=steps_per_second,filter_epsilon=filter_epsilon)
    @property
    def parameters(self):
        return create_hierarchical_Ox(filter(is_shared_parameter,self.variables),pi=len_parents(self))
    @property
    def inputs(self):
        return create_hierarchical_Ox(filter(is_input,self.variables),pi=len_parents(self))
    @property
    def states(self):
        return create_hierarchical_Ox(filter(is_state,self.variables),pi=len_parents(self))
    @property
    def variables(self):
        return create_hierarchical_Ox(theano_utils.get_named_variables_iter(self.outputs),pi=len_parents(self))


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


class ModelState(object):
    def __init__(self, d=None):
        if d is None:
            d = {}
        self.compute_state_dict = d
    def copy(self):
        return State(self.compute_state_dict.copy())
    
class Function(object):
    debug = False
    def __init__(self,model,outputs=None):
        """
            This class wraps a function created by Theano
            
            It remembers the order of inputs and outputs and plays well with
            the convis input and output dictionaries.
            
            `model` is a reference to the model and used for variable enumeration
            
            `outputs` can either be all outputs of the model, or additional terms or fewer.
        """
        self.model = model
        self.outputs = outputs
        self.givens = {}
        self.state = State()
        self.state_stack = []
    def pop_state(self):
        self.state = self.state_stack.pop()
    def push_state(self):
        self.state_stack.append(self.state.copy())
    @property
    def parameters(self):
        return create_hierarchical_Ox(filter(is_shared_parameter,theano_utils.get_named_variables_iter(self.outputs)),pi=len_parents(self))
    @property
    def inputs(self):
        return create_hierarchical_Ox(filter(is_input,theano_utils.get_named_variables_iter(self.outputs)),pi=len_parents(self))
    @property
    def states(self):
        return create_hierarchical_Ox(filter(is_state,theano_utils.get_named_variables_iter(self.outputs)),pi=len_parents(self))
    @property
    def variables(self):
        return create_hierarchical_Ox(theano_utils.get_named_variables_iter(self.outputs),pi=len_parents(self))
    def create_function(self,updates=None,additional_inputs=[],**kwargs):
        if self.debug is not True:
            # we disable warnings from theano gof because of the unresolved cache leak
            # TODO: fix the leak and re-enable warnings
            import logging
            logging.getLogger("theano.gof.cmodule").setLevel(logging.ERROR) 
        if updates is None:
            updates = theano.updates.OrderedUpdates()
        outputs = [o._as_TensorVariable() if hasattr(o,'_as_TensorVariable') else o for o in self.outputs]
        variables = unique_list([v for o in outputs for v in theano_utils.get_variables_iter(o)])
        for v in variables:
            if has_convis_attribute(v,'updates'):
                if v in updates:
                    updates[v] = T.mean([updates[v]]+[u for u in get_convis_attribute(v,'updates')],axis=0)
                else:
                    updates[v] = T.mean([u for u in get_convis_attribute(v,'updates')],axis=0)
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
        self.state.compute_state_dict = dict((v,None) for v in self.compute_output_order if is_out_state(v))
        if self.debug:# hasattr(retina, 'debug') and retina.debug:
            print('solving for:',outputs)
            print('all variables:',len(variables))
            print('input:',filter(is_input,variables))
            print('parameters:',filter(is_input_parameter,variables))
            print('states:',state_variables)
            print('updates:',self.compute_updates_order)
        self.reset_parameters()
        self.compute = theano.function(inputs=self.compute_input_order, 
                                    outputs=self.compute_output_order, 
                                    updates=self.compute_updates_order,
                                    givens=givens,on_unused_input='ignore',**kwargs)
        if self.debug is not True:
            import logging
            logging.getLogger("theano.gof.cmodule").setLevel(logging.WARNING) 
    def clear_states(self):
        self.state.compute_state_dict = {}
    def run(self,the_input,max_length,additional_inputs=[],inputs={},run_after=None,**kwargs):
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
                print('Chunk:', t, t+max_length)
            oo = self.run_single(the_input[t:(t+max_length)],
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
    def run_single(self,the_input,additional_inputs=[],inputs={},run_after=None,**kwargs):
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
            print("Using supplied inputs: ", inputs.keys())
        c = O()
        c.input = the_input
        c.model = self.model
        #c.config = self.config
        assert type(additional_inputs) == list, 'additional_inputs must be a list!'
        assert type(inputs) == dict, 'inputs must be a dictionary!'
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
                    input_dict[k] = get_convis_attribute(k,'param_init')(create_context_O(k,input=the_input,model=self.model,resolution=self.model.resolution))
                if is_state(k):
                    if self.state.compute_state_dict.get(get_convis_attribute(k,'state_out_state'),None) is None:
                        input_dict[k] = get_convis_attribute(k,'state_init')(create_context_O(k,input=the_input,model=self.model,resolution=self.model.resolution))
                    else:
                        input_dict[k] = self.state.compute_state_dict[get_convis_attribute(k,'state_out_state')]
        for shared_parameter in self.parameters._all:
            if is_shared_parameter(shared_parameter):
                #if (not hasattr(shared_parameter,'initialized') or shared_parameter.initialized == False) and not hasattr(shared_parameter,'optimized'):
                # until we can track which config values were changed, we re-initialize everything
                # all smart stuff is now in the injected .update method
                if (get_convis_attribute(shared_parameter,'initialized',False) is False) and not has_convis_attribute(shared_parameter,'optimized'):
                    get_convis_attribute(shared_parameter,'update')(create_context_O(shared_parameter,input=the_input,model=self.model,resolution=self.model.resolution))
                #shared_parameter.initialized = True
        the_vars = [input_dict[v] for v in self.compute_input_order]
        the_output = self.compute(*the_vars)
        if run_after is not None:
            run_after(the_output,self)
        for (o,k) in zip(the_output, self.compute_output_order):
            if is_out_state(k):
                self.state.compute_state_dict[k] = o
        return Output(the_output[:len(self.outputs)],keys=self.compute_output_order[:len(self.outputs)])


class Runner(object):
    def __init__(self, model=None, inputs=[], outputs=[]):
        self.model = model
        self.inputs = OrderedDict(inputs)
        self.outputs = OrderedDict(outputs)
        self.chunk_size = 100
        self.closed = True
    def stop(self):
        self.closed = True
    def start(self):
        if self.closed:
            import thread
            self.closed = False
            self.start_time = datetime.datetime.now()
            thread.start_new_thread(self.thread,tuple())
    def thread(self):
        import time, datetime
        while not self.closed:
            try:
                self.run()
            except StopIteration:
                time.sleep(0.5)
            time.sleep(0.1)
    def run(self, length = None, **kwargs):
        if length is not None:
            for t in xrange(0,length,self.chunk_size):
                o = self.model.run(chunk_size=self.chunk_size,
                                   **dict([(k,v) for k,v in self.inputs.items()]))
                for k in o.keys():
                    self.outputs[k].put(o[k])
            return o 
        else:
            o = self.model.run(**dict([(k,get_next(v)) for k,v in self.inputs.items()]))
            for k in o.keys():
                self.outputs[k].put(o[k])
            return o