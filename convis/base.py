"""
Convis base classes
----------------------

Convis extends PyTorch by adding some methods to `torch.nn.Module` and calling it a Layer.



"""
from __future__ import print_function
from .misc_utils import unique_list, suppress

import numpy as np
import matplotlib.pylab as plt
import uuid
from . import io
try:
    from exceptions import NotImplementedError
except ImportError:
    pass
from .variable_describe import describe, describe_dict, describe_html
import warnings

from . import variables
from .variables import *
from . import utils
from . import o
from . import optimizer
from .o import O, Ox, save_name
from collections import OrderedDict

# ----

import torch
import numpy as np
from torch import nn
import datetime

try:
    from functools import reduce
except:
    pass

from .variables import Variable, State, Parameter, as_parameter, is_variable
from copy import deepcopy

TIME_DIMENSION = 2

### Node and Model classes

def len_parents(n):
    if hasattr(n,'parent') and n.parent != n:
        return len_parents(n.parent)+1
    return 0

class Output(object):
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
    def __init__(self,outs,keys=None):
        """
            The full path names of all variables are also added to this objects __dict__,
            allowing for tab completion.

            By default returns a numpy array when used with square brackets, the 
            Variable when accessed over `._outs[n]`.
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
        self.__dict__.update(self._out_dict)
    def __len__(self):
        return len(self._outs)
    def __iter__(self):
        return iter(self._outs)
    def plot(self,k=0):
        utils.plot(self[k])
    def array(self,k=0):
        return np.array(self[k])
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

class _OptimizerSelection(object):
    """
        A single optimizer option that can be called to set this optimizer for the model.
        The doc string of the original optimizer is available by calling `help()` on this object.
    """
    def __init__(self, model, opt):
        self._model = model
        self._opt = opt
        self.__doc__ = self._opt.__doc__
    def __call__(self,*args,**kwargs):
        self._model.set_optimizer(self._opt,*args,**kwargs)

class _OptimizerSelector(object):
    """
        Enables tab completion to set optimizers for a model.
        Includes all Optimizers found in `torch.nn.optim` and
        `convis.optimizer`.

        If optimizers are added to torch during runtime,
        you can call `._reload()` to add all available options.
    """
    def __init__(self, model):
        self._model = model
        for o in dir(torch.optim):
            if type(getattr(torch.optim, o)) is type and issubclass(getattr(torch.optim, o),torch.optim.Optimizer):
                setattr(self, o, _OptimizerSelection(self._model,getattr(torch.optim, o)))
        for o in dir(optimizer):
            if type(getattr(optimizer, o)) is type and issubclass(getattr(optimizer, o),torch.optim.Optimizer):
                setattr(self, o, _OptimizerSelection(self._model,getattr(optimizer, o)))
    def _reload(self):
        for o in dir(torch.optim):
            if type(getattr(torch.optim, o)) is type and issubclass(getattr(torch.optim, o),torch.optim.Optimizer):
                setattr(self, o, _OptimizerSelection(self._model,getattr(torch.optim, o)))
        for o in dir(optimizer):
            if type(getattr(optimizer, o)) is type and issubclass(getattr(optimizer, o),torch.optim.Optimizer):
                setattr(self, o, _OptimizerSelection(self._model,getattr(optimizer, o)))
    def __call__(self, opt, *args, **kwargs):
        import types
        if len(args) == 0 or (type(args[0]) is not list and not isinstance(args[0], types.GeneratorType)):
            args = list(args)
            args.insert(0,self._model.parameters())
        if issubclass(opt,torch.optim.Optimizer):
            self._model._optimizer = opt(*args,**kwargs)
        if type(opt) is str and issubclass(getattr(torch.optim, opt),torch.optim.Optimizer):
            self(getattr(torch.optim, opt),*args,**kwargs)

def prepare_input(a, dims= 5, cuda=False, volatile=False, requires_grad = False):
    """
        Utility function to broadcast input to 5 dimensions, make it a Tensor,
        wrap it in a Variable and optionally move it to the GPU.

        Short hand for::

            import torch
            a_var = torch.autograd.Variable(torch.Tensor(a[None,None,:,:,:]), requires_grad=True).cuda()

            from convis.base import prepare_input
            a_var = prepare_input(a, cuda=True, requires_grad=True)

    """
    if not type(a) is torch.autograd.Variable:
        if hasattr(a, 'numpy'):
            # its hopefully a torch.Tensor
            a = torch.autograd.Variable(a, volatile=volatile, requires_grad=requires_grad)
        else:
            a = torch.autograd.Variable(torch.Tensor(a), volatile=volatile, requires_grad=requires_grad)
    if dims is not None:
        if dims == 5:
            if len(a.data.shape) == 3:
                a = a[None,None,:,:,:]
        if dims == 3:
            if len(a.data.shape) == 5:
                a = a[0,0,:,:,:]
    if cuda is not None:
        if cuda:
            return a.cuda()
        else:
            return a.cpu()

def shape(x):
    """
        Return the shape of a Tensor or Variable containing a Tensor.
    """
    if hasattr(x,'shape'):
        return x.shape
    if hasattr(x,'data'):
        if hasattr(x.data,'shape'):
            return x.data.shape
    raise Exception('No shape found for '+str(x)+'!')

class Layer(torch.nn.Module):
    """
        Base class for modules, layers and models.

        `convis.Layer` is a `torch.nn.Module` with some added functionality::

            import convis
            import torch.nn.functional as F

            class Model(convis.Layer):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = convis.filters.Conv3d(1, (20,1,1))
                    self.conv2 = convis.filters.Conv3d(1, (1,10,10))
                def forward(self, x):
                   x = F.relu(self.conv1(x))
                   return F.relu(self.conv2(x))

        Just as `Module`s, `Layer`s can include other `Layer`s or `Module`s (ie. its `sublayers`).
        `Variable`s, `Parameter`s and `State`s that are attributes of a Layer or
        its `sublayers` will be registered and can be collected according to their
        class.
        
        All registered Variables (including Parameters and States), will be moved to the
        corresponding device when calling `.cuda()` or `.cpu()`.

        In contrast to many methods of torch.Tensors, Layer methods are always
        in-place! Using `.cuda()` or `.float()` will return a reference to the 
        original model and not a copy.


        Attributes
        ----------

        _use_cuda : bool
        set_optimizer : _OptimizerSelector objet that allows tab completion to select an optimizer

        Methods
        -------

        cuda(device=None)
            move the model to the gpu
        cpu()
            move the model to the cpu

        run(the_input, dt=None)
            execute the model, using chunk sizes of `dt`

        parse_config(conf)

        optimize(inp,outp,dt=None)
            use the selected optimizer to fit the model
            to return outp to the input inp.
            Accepts a chunk length `dt`

        register_state(name, value)
            registers an attribute name to be a state variable
        
        state()
            returns the current state of the model
            (recursively for all submodules)

        set_state(d)
            set all state parameters defined in dictionary `d`
            to the corresponding values.

        push_state()
            pushes the current state on a stack

        pop_state()
            pops the last state from the stack
            and sets all state variables to the 
            corresponding values.


        Examples
        --------


        >>> import convis
        >>> import torch.nn.functional as F
        >>> 
        >>> class Model(convis.Layer):
        >>>     def __init__(self):
        >>>         super(Model, self).__init__()
        >>>         self.conv1 = convis.filters.Conv3d(1, (20,1,1))
        >>>         self.conv2 = convis.filters.Conv3d(1, (1,10,10))
        >>>     def forward(self, x):
        >>>        x = F.relu(self.conv1(x))
        >>>        return F.relu(self.conv2(x))


        See Also
        --------

        torch.nn.Module : torchs layer class
    """
    _state = OrderedDict()
    def __init__(self):
        self._state = OrderedDict()
        self._default_state = OrderedDict()
        super(Layer, self).__init__()
        self._variables = []
        self._named_variables= {}
        self._use_cuda = False
        self._optimizer = None
        self.set_optimizer = _OptimizerSelector(self)
    def register_state(self,name,val=None):
        if hasattr(self,name) and val is None:
            self._state[name] = getattr(self,name)
        else:
            self._state[name] = val
        self._default_state[name] = deepcopy(val)
    def cuda(self, device=None):
        """
            Moves the model to the GPU (optionally with number `device`).
            returns the model itself.
        """
        self._use_cuda = True
        for m in self.modules():
            if m is not self:
                m.cuda()
        if device is not None:
            return super(Layer, self).cuda(device=device)
        else:
            return super(Layer, self).cuda()
    def cpu(self):
        """
            Moves the model to the CPU and returns the model itself.
        """
        self._use_cuda = False
        for m in self.modules():
            if m is not self:
                m.cpu()
        return super(Layer, self).cpu()
    def __call__(self,*args,**kwargs):
        new_args = []
        for a in args:
            a = prepare_input(a, dims=getattr(self,'dims',None), cuda=self._use_cuda)
            new_args.append(a)
        o0 = o = super(Layer, self).__call__(*new_args,**kwargs)
        if hasattr(self, 'outputs'):
            o = Output([o] + [getattr(self,k) for k in self.outputs], keys = ['output']+self.outputs)
        return o
    def run(self,the_input,dt=None,t=0):
        """
            Runs the model either once, or multiple times to process chunks of size `dt`.

            Returns an `Output` object.
        """
        if dt is not None:
            return self._run_in_chunks(the_input,dt=dt,t=t)
        else:
            return self(the_input)
    def _run_in_chunks(self,the_input,dt=100,t=0):
        chunked_output = []
        keys = ['output']
        if len(shape(the_input)) == 3:
            the_input = the_input[None,None,:,:,:]
        while t < shape(the_input)[2]:
            oo = self(the_input[:,:,t:(t+dt),:,:])
            if type(oo) is not Output:
                oo = [oo]
            else:
                keys=oo.keys
            for i,o in enumerate(oo):
                while len(chunked_output) < i+1:
                    chunked_output.append([])
                chunked_output[i].append(o)
            t += dt
        outs = []
        for co in chunked_output:
            try:
                outs.append(torch.cat(co,dim=TIME_DIMENSION))
            except Exception as e:
                outs.append(np.array(co))
        return Output(outs,keys=keys)
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
    def __dir__(self):
        if hasattr(self,'_state'):
            return self._state.keys() + self.__dict__.keys() + super(Layer, self).__dir__()
        return self.__dict__.keys() + super(Layer, self).__dir__()
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(Layer, self).__setattr__(name, value)
        if name in self._state.keys():
            self._state[name] = value
        if is_variable(value):
            if type(value) == Parameter:
                self._parameters[name] = value
            self._variables.append(value)
            self._named_variables[name] = value
            self.__dict__[name] = value
        else:
            super(Layer, self).__setattr__(name, value)
    def __getattr__(self, name):
        if name.startswith('_'):
            return super(Layer, self).__getattr__(name)
        if name in self._state.keys():
            return self._state[name]
        else:
            return super(Layer, self).__getattr__(name)
    def parse_config(self,config,prefix='',key='retina_config_key'):
        """
            Loads parameter values from a configuration (RetinaConfiguration or dict).
        """
        def f(a):
            if hasattr(a,'_variables'):
                for v in a._variables:
                    if hasattr(v,'retina_config_key'):
                        if v.retina_config_key.startswith('--'):
                            continue
                        if hasattr(v,'set'):
                            try:
                                v.set(config.get(prefix+v.retina_config_key))
                            except:
                                pass
                        else:
                            print('has no set:',v)
        self.apply(f)
    def optimize(self, inp, outp, optimizer = None, loss_fn = lambda x,y: ((x-y)**2).sum(), dt=None, t=0):
        """
            Runs an Optimizer to fit the models parameters such that the output
            of the model when presented `inp` approximates `outp`.

            To use this function, an Optimizer has to be selected::

                model.set_optimizer(torch.optim.SGD(lr=0.01))
                model.optimize(x,y, dt=100)

            or::

                model.set_optimizer.SGD(lr=0.01) # uses optimizers from torch.optim
                model.optimize(x,y, dt=100)

            It is important to specify a chunk length `dt`, if the complete input does not fit into memory.

        """
        if optimizer is None:
            assert self._optimizer is not None, 'Optimizer has to be set! Use .set_optimizer.<tab>'
            optimizer = self._optimizer
        def closure():
            self.pop_state()
            self.push_state()
            optimizer.zero_grad()
            o = self(closure.inp)
            loss = closure.loss_fn(closure.outp,o)
            loss.backward(retain_graph=True)
            return loss
        inp = prepare_input(inp, dims=getattr(self,'dims',None), cuda=self._use_cuda)
        outp = prepare_input(outp, dims=getattr(self,'dims',None), cuda=self._use_cuda)
        self.push_state()
        if dt is not None:
            steps = []
            while t < shape(inp)[TIME_DIMENSION]:
                closure.inp = inp[:,:,t:(t+dt),:,:]
                closure.outp = outp[:,:,t:(t+dt),:,:]
                t += dt
                closure.loss_fn = loss_fn
                steps.append(optimizer.step(closure))
            return steps
        else:
            closure.inp = inp
            closure.outp = outp
            closure.loss_fn = loss_fn
            return optimizer.step(closure)
    def state(self):
        """
            collects the state and returns an
            OrderedDict 
        """
        def rec(model):
            o = OrderedDict()
            if hasattr(model,'_state'):
                for s_name in model._state.keys():
                    o[s_name] = getattr(model, s_name)
            for mod_name,mod in list(model.named_modules()):
                if mod is model:
                    continue
                for s_name,s in rec(mod).items():
                    o[mod_name+'.'+s_name] = s
            return o
        return rec(self)
    def set_state(self, state_dict):
        """
            collects the state and returns an
            OrderedDict 
        """
        def rec(model, sub_state_dict):
            if hasattr(model,'_state'):
                for s_name,s in model._state.items():
                    setattr(model, s_name, sub_state_dict[s_name])
            for mod_name,mod in list(model.named_modules()):
                if mod is model:
                    continue
                new_sub_state_dict = OrderedDict()
                for s_name,s in sub_state_dict.items():
                    if s_name.startswith(mod_name+'.'):
                        new_sub_state_dict[s_name[len(mod_name+'.'):]] = s
                rec(mod, new_sub_state_dict)
        return rec(self, state_dict)
    def clear_state(self):
        """
            resets the state to default values
        """
        def rec(model):
            o = OrderedDict()
            if hasattr(model,'_state'):
                for s_name,s in model._state.items():
                    if hasattr(model,'_default_state'):
                        old_val = model._state.get(s_name, None)
                        setattr(model,s_name, deepcopy(model._default_state.get(s_name, None)))
                    else:
                        old_val = model._state.get(s_name, None)
                        setattr(model,s_name, None)
                    o[s_name] = (old_val,getattr(model, s_name, None))
            for mod_name,mod in list(model.named_modules()):
                if mod is model:
                    continue
                for s_name,s in rec(mod).items():
                    o[mod_name+'.'+s_name] = s
            return o
        return rec(self)
    def push_state(self):
        """
            collects all State variables and pushes they values onto a stack
        """
        if not hasattr(self, '_state_stack'):
            self._state_stack = []
        self._state_stack.append(self.state())
    def pop_state(self):
        """
            retrieves the values of all State variables and from a stack
        """
        if not hasattr(self, '_state_stack'):
            raise Exception('No state was pushed to the stack!')
        self.set_state(self._state_stack.pop())

Model = Layer

from collections import OrderedDict
import time

def get_next(stream,l=100):
    if hasattr(stream, 'get'):
        return stream.get(l)
    elif hasattr(stream, '__next__'):
        return next(stream)
    elif type(stream) == list:
        if len(stream) > 0:
            return stream.pop(0)
        else:
            raise StopIteration()
    else:
        return stream

class _DummyModel(object):
    def __init__(self):
        pass
    def run(self, chunk_size=20, **kwargs):
        inps = dict([(k,get_next(v,l=chunk_size)*np.random.rand()) for k,v in kwargs.items()])
        return Output(inps.values(),keys=inps.keys())
        
class Runner(object):
    """
        Keeps track of the input and output of a model
        and can run or optimize it in a separate thread.

        `model` has to be a `convis.Layer`

        `input` should be a `convis.streams.Stream` that contains input data
        `output` should be a `convis.streams.Stream` that accepts new data
        when using optimize, `goal` has to have the same length as input and the same behaviour at the end of the stream (repeating or stop)

    """
    def __init__(self, model=None, input=None, output=None, goal=None):
        self.model = model
        self.input = input
        self.output = output
        self.goal = goal
        self.chunk_size = 20
        self.closed = True
    def stop(self):
        self.closed = True
    def start(self):
        if self.closed:
            try:
                import thread
            except ImportError:
                import _thread as thread
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
                o = self.model.run(self.input, dt=self.chunk_size)
                self.output.put(o.data.cpu().numpy()[0,0])
            return o 
        else:
            o = self.model.run(get_next(self.input))
            if hasattr(o,'keys'):
                self.output.put(o[0].data.cpu().numpy()[0,0])
            else:
                self.output.put(o.data.cpu().numpy()[0,0])
            return o
    def optimize(self):
        o = self.model.optimize(get_next(self.input), get_next(self.goal))
        if hasattr(o,'keys'):
            self.output.put(o[0].data.cpu().numpy()[0,0])
        else:
            self.output.put(o.data.cpu().numpy()[0,0])
        return o
