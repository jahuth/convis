import litus
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pylab as plt
from theano.tensor.nnet.conv3d2d import conv3d
from theano.tensor.signal.conv import conv2d
import uuid
from . import retina_base
from exceptions import NotImplementedError

def f7(seq):
    """ This function is removing duplicates from a list while keeping the order """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

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


def get_inputs(apply_node):
    """ get variables that have no owner """
    if apply_node is None or not hasattr(apply_node,'owner'):
        return []
    if apply_node.owner is None:
        return [apply_node]
    inputs = apply_node.owner.inputs
    parent_inputs = []
    for i in inputs:
        if hasattr(i,'owner'):
            parent_inputs += get_inputs(i)
    return parent_inputs

def get_named_variables(apply_node):
    """ get variables that have a name """
    if apply_node is None or not hasattr(apply_node,'owner'):
        return []
    if apply_node.name is None:
        parent_inputs = []
    else:
        parent_inputs = [apply_node]
    if apply_node.owner is None:
        return parent_inputs
    inputs = apply_node.owner.inputs
    for i in inputs:
        parent_inputs += get_named_variables(i)
    return parent_inputs

def get_all_variables(apply_node):
    """ get all variables that are parents of the specific graph node """
    if apply_node is None or not hasattr(apply_node,'owner'):
        return []
    parent_inputs = [apply_node]
    if apply_node.owner is None:
        return parent_inputs
    inputs = apply_node.owner.inputs
    for i in inputs:
        parent_inputs += get_all_variables(i)
    return parent_inputs

def get_variables_iter(apply_node,depth=None,ignore=[],explore_scan=True,include_copies=False):
    """ get variables that have a name """
    if hasattr(apply_node,'_as_TensorVariable'):
        apply_node = apply_node._as_TensorVariable() 
    if type(apply_node) in [list,tuple]:
        return f7([b for a in apply_node for b in get_variables_iter(a,depth=depth)])
    else:
        nodes_to_explore = [apply_node]
        nodes_explored = []
        while len(nodes_to_explore) > 0:
            node = nodes_to_explore.pop()
            if node in nodes_explored or node in ignore:
                # easier to comprehend than checking when adding
                continue
            nodes_explored.append(node)
            if hasattr(node,'owner') and node.owner is not None:
                nodes_to_explore.extend([i for i in node.owner.inputs if not i in nodes_explored and not i in ignore])
            if hasattr(node,'state_out_state') and node.state_out_state not in nodes_explored:
                nodes_to_explore.append(node.state_out_state)
            if hasattr(node,'copied_from') and node.copied_from not in nodes_explored:
                nodes_to_explore.append(node.copied_from)
            if explore_scan and is_scan_op(node):
                nodes_to_explore.extend(node.owner.op.outputs)
            if depth is not None and len(nodes_explored) > depth:
                break
        return nodes_explored

def get_input_variables_iter(apply_node,depth=None,ignore=[],explore_scan=True,include_copies=False):
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
        elif hasattr(node,'copied_from'):
            nodes_to_explore.append(node.copied_from)
        else:
            input_nodes.append(node)
        if depth is not None and len(nodes_explored) > depth:
            break
    return input_nodes

def get_named_variables_iter(apply_node,depth=None,ignore=[],explore_scan=True,include_copies=False):
    """ get variables that have a name """
    return filter(lambda x: x.name is not None and hasattr(x,'__is_convis_var'), get_variables_iter(apply_node,depth=depth,ignore=ignore,explore_scan=explore_scan,include_copies=include_copies))
def get_named_input_variables_iter(apply_node,depth=None,ignore=[],explore_scan=True,include_copies=False):
    """ get variables that have a name """
    return filter(lambda x: x.name is not None and hasattr(x,'__is_convis_var'), get_input_variables_iter(apply_node,depth=depth,ignore=ignore,explore_scan=explore_scan,include_copies=include_copies))

#def replace(apply_node,old,new):
#    apply_node.replace(old,new)

def replace(apply_node,old,new,depth=None):
    for v in get_variables_iter(apply_node,depth=depth):
        try:
            #for old in v.owner.inputs:
            #    #v.owner.inputs.replace(old,new)
            inputs = v.owner.inputs
            v.owner.inputs = map(lambda x:new if x==old else x,inputs)
        except:
            pass

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


def is_scan_op(n):
    if hasattr(n,'owner') and hasattr(n.owner,'op') and type(n.owner.op) == theano.scan_module.scan_op.Scan:
        return True
    return False