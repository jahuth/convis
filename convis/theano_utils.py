import theano
import theano.tensor as T


try:
    from theano.tensor.nnet import conv3d as theano_conv3d
    from theano.tensor.nnet import conv2d as theano_conv2d

    def conv2d(inp,filt,*args,**kwargs):
        """
            Converts the new (>Theano 9.0) conv2d interface to work with a 3d image and a 2d kernel.
        """
        if inp.ndim == 2 and filt.ndim == 2:
            return theano_conv2d(inp.dimshuffle('x','x',0,1),filt.dimshuffle('x','x',0,1),*args,**kwargs)[0,0,:,:]
        elif inp.ndim == 3 and filt.ndim == 2:
            return theano_conv2d(inp.dimshuffle(0,'x',1,2),filt.dimshuffle('x','x',0,1),*args,**kwargs)[:,0,:,:]
        else:
            return theano_conv2d(inp,filt,*args,**kwargs)[:,0,:,:]

    def conv3d(inp,filt,*args,**kwargs):
        return theano_conv3d(inp.dimshuffle(0,2,1,3,4),filt.dimshuffle(0,2,1,3,4),*args,**kwargs).dimshuffle(0,2,1,3,4)
except ImportError:
    # old convolutions
    from theano.tensor.nnet.conv3d2d import conv3d
    from theano.tensor.signal.conv import conv2d


import numpy as np
import matplotlib.pylab as plt
import uuid
from . import retina_base
from misc_utils import unique_list, suppress
from exceptions import NotImplementedError
from variables import get_convis_attribute, has_convis_attribute, set_convis_attribute, Variable, is_var


dtensor5 = T.TensorType('float64', (False,)*5)

_nd_conversions = {
    0: {
        0: lambda x: x,
        1: lambda x: x.dimshuffle(('x')),
        2: lambda x: x.dimshuffle(('x','x')),
        3: lambda x: x.dimshuffle(('x','x','x')),
        4: lambda x: x.dimshuffle(('x','x','x','x')),
        5: lambda x: x.dimshuffle(('x','x','x','x','x'))
    },
    1: {
        1: lambda x: x,
        2: "Can not convert time to space.",
        3: lambda x: x.dimshuffle((0,'x','x')),
        4: "Can not convert to 4d.",
        5: lambda x: x.dimshuffle(('x',0,'x','x','x'))        
    },
    2: {
        1: "Can not convert space to time.",
        2: lambda x: x,
        3: lambda x: x.dimshuffle(('x',0,1)),
        4: "Can not convert to 4d.",
        5: lambda x: x.dimshuffle(('x','x','x',0,1))        
    },
    3: {
        1: lambda x: x.dimshuffle((0)),
        2: lambda x: x.dimshuffle((1,2)),
        3: lambda x: x,
        4: "Can not convert to 4d.",
        5: lambda x: x.dimshuffle(('x',0,'x',1,2))        
    },
    4: {},
    5: {
        1: lambda x: x.dimshuffle((1)),
        2: lambda x: x.dimshuffle((3,4)),
        3: lambda x: x[0,:,0,:,:],
        4: "Can not convert to 4d.",
        5: lambda x: x
    }
}
def make_nd(inp,dim=3):
    """
    This function reshapes 1d, 2d, 3d and 5d tensor variables into each other under the following assumptions:
    
      * a 1d tensor contains only timeseries data
      * a 2d tensor contains only spatial data
      * a 3d tensor has time as the first dimension, space as second and third
      * a 5d tensor has the dimensions (0,time,0,x,y), where 0 is an empty dimension
      
    When the input tensor already has the desired number of dimensions, it is returned.
    """
    if hasattr(inp,'_as_TensorVariable'):
        inp = inp._as_TensorVariable() 
    from_d = inp.ndim
    f = _nd_conversions.get(from_d,{}).get(dim,"No valid conversion found.")
    if type(f) is str:
        raise Exception(f)
    return f(inp)

def conv3(inp,kernel):
    """
        Convolves 3d sequence with kernel (which normally have to be cast to 5d tensors).
    """
    return make_nd(conv3d(make_nd(inp,5),make_nd(kernel,5)),3)

### helper functions on theano variable graphs

# recursive graph traversal is deprecated
# def get_inputs(apply_node):
#     """ get variables that have no owner """
#     if apply_node is None or not hasattr(apply_node,'owner'):
#         return []
#     if apply_node.owner is None:
#         return [apply_node]
#     inputs = apply_node.owner.inputs
#     parent_inputs = []
#     for i in inputs:
#         if hasattr(i,'owner'):
#             parent_inputs += get_inputs(i)
#     return parent_inputs

# def get_named_variables(apply_node):
#     """ get variables that have a name """
#     if apply_node is None or not hasattr(apply_node,'owner'):
#         return []
#     if apply_node.name is None:
#         parent_inputs = []
#     else:
#         parent_inputs = [apply_node]
#     if apply_node.owner is None:
#         return parent_inputs
#     inputs = apply_node.owner.inputs
#     for i in inputs:
#         parent_inputs += get_named_variables(i)
#     return parent_inputs

# def get_all_variables(apply_node):
#     """ get all variables that are parents of the specific graph node """
#     if apply_node is None or not hasattr(apply_node,'owner'):
#         return []
#     parent_inputs = [apply_node]
#     if apply_node.owner is None:
#         return parent_inputs
#     inputs = apply_node.owner.inputs
#     for i in inputs:
#         parent_inputs += get_all_variables(i)
#     return parent_inputs

def get_variables_iter(apply_node,depth=None,ignore=[],explore_scan=True,include_copies=False,factory=None):
    """ get variables that have a name """
    if hasattr(apply_node,'_as_TensorVariable'):
        apply_node = apply_node._as_TensorVariable() 
    if type(apply_node) in [list,tuple]:
        return unique_list([b for a in apply_node for b in get_variables_iter(a,depth=depth)])
    else:
        nodes_to_explore = [apply_node]
        nodes_explored = []
        while len(nodes_to_explore) > 0:
            node = nodes_to_explore.pop()
            if node in nodes_explored or node in ignore:
                # easier to comprehend than checking when adding
                continue
            if node is not None:
                nodes_explored.append(node)
            if hasattr(node,'owner') and node.owner is not None:
                nodes_to_explore.extend([i for i in node.owner.inputs if not i in nodes_explored and not i in ignore])
            if has_convis_attribute(node,'state_out_state') and get_convis_attribute(node,'state_out_state') not in nodes_explored:
                nodes_to_explore.append(get_convis_attribute(node,'state_out_state'))
            if has_convis_attribute(node,'copied_from') and get_convis_attribute(node,'copied_from') not in nodes_explored:
                nodes_to_explore.append(get_convis_attribute(node,'copied_from'))
            if explore_scan and is_scan_op(node):
                nodes_to_explore.extend(node.owner.op.outputs)
            if depth is not None and len(nodes_explored) > depth:
                break
        if factory is not None:
            return [factory(n) for n in nodes_explored]
        return nodes_explored

def get_input_variables_iter(apply_node,depth=None,ignore=[],explore_scan=True,include_copies=False,factory=None):
    """ get variables that have a name """
    if hasattr(apply_node,'_as_TensorVariable'):
        apply_node = apply_node._as_TensorVariable() 
    nodes_to_explore = [apply_node]
    input_nodes = []
    nodes_explored = []
    while len(nodes_to_explore) > 0:
        node = nodes_to_explore.pop()
        if node in nodes_explored or node in ignore:
            # easier to comprehend than checking when adding
            continue
        nodes_explored.append(node)
        if hasattr(node,'owner') and node.owner is not None and len(node.owner.inputs) > 0:
            nodes_to_explore.extend([i for i in node.owner.inputs if not i in nodes_explored and not i in ignore])
        elif explore_scan and is_scan_op(node):
            nodes_to_explore.extend(node.owner.op.outputs)
        elif has_convis_attribute(node,'copied_from'):
            nodes_to_explore.append(get_convis_attribute(node,'copied_from'))
        else:
            input_nodes.append(node)
        if depth is not None and len(nodes_explored) > depth:
            break
    if factory is not None:
        return [factory(n) for n in input_nodes]
    return input_nodes

def get_named_variables_iter(apply_node,depth=None,ignore=[],explore_scan=True,include_copies=False,factory=None):
    """ get variables that have a name """
    return filter(lambda x: get_convis_attribute(x,'name',None) is not None and is_var(x), 
                    get_variables_iter(apply_node,
                        depth=depth,
                        ignore=ignore,
                        explore_scan=explore_scan,
                        include_copies=include_copies,
                        factory=factory))
def get_named_input_variables_iter(apply_node,depth=None,ignore=[],explore_scan=True,include_copies=False,factory=None):
    """ get variables that have a name """
    return filter(lambda x: get_convis_attribute(x,'name',None) is not None and is_var(x), 
                    get_input_variables_iter(apply_node,
                        depth=depth,
                        ignore=ignore,
                        explore_scan=explore_scan,
                        include_copies=include_copies,
                        factory=factory))

#def replace(apply_node,old,new):
#    apply_node.replace(old,new)

def replace(apply_node,old,new,depth=None):
    for v in get_variables_iter(apply_node,depth=depth):
        with suppress():
            if v.owner is not None:
                inputs = v.owner.inputs
                v.owner.inputs = map(lambda x:new if x==old else x,inputs)

def _replace(apply_node,old,new,depth=300):
    """ replaces one variable in a graph with another one """
    if depth == 0:
        print "Reached max depth!"
        return
    if apply_node is None or not hasattr(apply_node,'owner'):
        return
    if apply_node.owner is None:
        return
    inputs = apply_node.owner.inputs
    for i in inputs:
        if i == old:
            print apply_node.owner.inputs
            print ">>> changing ",old,"to",new
            apply_node.owner.inputs = map(lambda x:new if x==old else x,inputs)
            print apply_node.owner.inputs
        else:
            _replace(i,old,new,depth-1)
    return

def add_input_to_a_sum_op(sum_op, v, replace_inputs = True):
    """

        This function takes an ApplyNode of a Sum Op and adds a new input to it.

    """
    if type(sum_op.owner.op) == T.elemwise.Sum and hasattr(sum_op.owner,'inputs') and len(sum_op.owner.inputs) > 0:
        # Since Theano 9.0 T.sum([a]) will no longer give a Join.0([a]), but a InplaceDimShuffle{x,0,1,2}(a)
        if isinstance(sum_op.owner.inputs[0].owner.op, theano.tensor.DimShuffle):
            # we replace the DimShuffle with a one element Join that can then be extended
            sum_op.owner.inputs[0] = T.Join()(0,sum_op.owner.inputs[0].owner.inputs)
        if isinstance(sum_op.owner.inputs[0].owner.op, theano.tensor.Join):
            # The Sum Op contains a Join as an input.
            # If the dimensions match, we extend the Join with another element (the new input).
            # A special case is the first new input:
            #  if the previous input was a zero tensor with a 'replaceable_input' flag,
            #  this tensor will be removed from the list when an input is provided.
            if sum_op.owner.inputs[0].owner.inputs[1].ndim == 3+1:
                # assuming a 3d input/ output
                if replace_inputs and has_convis_attribute(sum_op.owner.inputs[0].owner.inputs[1].owner.inputs[0],'replaceable_input'):
                    sum_op.owner.inputs[0].owner.inputs[1] = make_nd(v,3).dimshuffle(('x',0,1,2)) # TODO: We add a dimension for summing? Don't we have that from the list?
                else:
                    sum_op.owner.inputs[0].owner.inputs.append(make_nd(v,3).dimshuffle(('x',0,1,2)))
                return True
            elif sum_op.owner.inputs[0].owner.inputs[1].ndim == 5+1:
                # otherwise a 5 dimensional input
                if replace_inputs and has_convis_attribute(sum_op.owner.inputs[0].owner.inputs[1].owner.inputs[0],'replaceable_input'):
                    sum_op.owner.inputs[0].owner.inputs[1] = make_nd(v,5).dimshuffle(('x',0,1,2,3,4))
                else:
                    sum_op.owner.inputs[0].owner.inputs.append(make_nd(v,5).dimshuffle(('x',0,1,2,3,4)))
                return True
            else:
                raise Exception('Specified input is a sum of tensors that neither have 3 or 5 dimensions!')
        else:
            # Since a change in Theano broke the input mechanics without any usefull error,
            # the user should at least know that this part of the toolbox depends on the Theano
            # version and can break when Theano changes how sums are handled.
            raise Exception('The Sum Op has neither a Join nor a DimShuffle as an input! This means that your version of Theano is possibly too new (try 8.2 or 9.0).')
    else:
        return False

def is_scan_op(n):
    if hasattr(n,'owner') and hasattr(n.owner,'op') and type(n.owner.op) == theano.scan_module.scan_op.Scan:
        return True
    return False


## padding subgraphs

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
    ar = make_nd(ar,5)
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
    return make_nd(pad5_txy(make_nd(ar,5),Nt,Nx,Ny,mode=mode,c=c),3)

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
    return make_nd(pad5(make_nd(ar,5),N=N,axis=axis,mode=mode,c=c),2)

def pad2_xy(ar,Nx,Ny,mode='mirror',c=0.0):
    """
        Padds a 2 dimensional tensor with `Nx` and `Ny` bins in x and y direction.
        If the tensor does not have 2 dimensions, it will be converted.
        Returns a 2 dimensional tensor.

        see `pad5_txy` and `pad5`
    """
    return make_nd(pad5_txy(make_nd(ar,5),0,Nx,Ny,mode=mode,c=c),2)
