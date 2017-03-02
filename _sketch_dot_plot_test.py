"""

Exploring the whole graph is a bit tricky.


One problem are scan ops. They create a sub-graph that only contains copies of the variables declared supplied to the scan.

ScanOp Apply Node:
    inputs = the list of inputs to the scan op
    outputs = the list of outputs from the scan op
    op =
        ScanOp:
            outputs = the clones of the outputs that give access to the internal graph

It is easy to just bypass the scan op by treating it as a normal Op.
Then it will just be a box that gets the proper in- and outputs.

But if we want to also show the internals of the box, we have to be creative.

When a graph is wrapped in a GraphWrapper, 



To make variables copyable by theano I overwrote the copy function of the variable and 
cloned the type and injected functions into there (__call__ and make_variable) which
copy all annotations over to the new variable.
Also the original variable is referenced.

"""
import convis
from convis.o import save_name
from convis.variables import get_convis_attribute, has_convis_attribute, set_convis_attribute, full_path, is_var, is_named_var
config = convis.retina.RetinaConfiguration()
retina = convis.retina.Retina(config)
retina.outputs.append(retina.opl.graph)
all_vars = convis.theano_utils.get_variables_iter(retina.outputs, explore_scan = True)

def is_scan_op(n):
    if hasattr(n,'owner') and hasattr(n.owner,'op') and type(n.owner.op) == convis.theano.scan_module.scan_op.Scan:
        return True
    return False

ops_as_node ={
    'Elemwise{switch,no_inplace}': 'if %s then %s else %s',
    'Elemwise{lt,no_inplace}':'(%s < %s)',
    'Elemwise{gt,no_inplace}':'(%s > %s)',    
    'Elemwise{int_div,no_inplace}': '%s / %s',
    'Elemwise{true_div,no_inplace}': '%s / %s',
    'Elemwise{pow,no_inplace}':'%s**(%s)',
    'Elemwise{sub,no_inplace}':'(%s - %s)',
    'Elemwise{mul,no_inplace}':'%s * %s',
    'Elemwise{add,no_inplace}':'(%s + %s)',
    'Elemwise{exp,no_inplace}':'exp(%s)',
    'DimShuffle': '%s',
    'Subtensor': '%s',
    'AbstractConv2d': 'conv2d(%s, %s)',
    'ConvOp': 'conv3d(%s, %s)'
}

def build_formula(node,ignore_name=False,rec_counter=10):
    if rec_counter < 0:
        return '...'
    try:
        return str(node.value.tolist())
    except:
        pass
    if is_named_var(node) and not ignore_name:
        return get_convis_attribute(node,'html_name',get_convis_attribute(node,'name'))
    op = ''
    owner_op = node.owner.op
    if hasattr(owner_op,'name') and owner_op.name in ops_as_node.keys():
        op = ops_as_node[owner_op.name]
    try:
       op = ops_as_node[owner_op.__class__.__name__]
    except:
        pass
    inputs = []
    for i in node.owner.inputs:
        inputs.append(build_formula(i,False,rec_counter-1))
    if op == '':
        return ','.join(inputs)
    return op%tuple(inputs[:(len(op.split('%s'))-1)])

var_counter = 1
# def full_path(v):
#     if type(v) == str:
#         return save_name(v)
#     if hasattr(v,'copied_from'):
#         return full_path(v.copied_from)
#     if not hasattr(v,'path') or v.path is None:
#         if not hasattr(v,'name') or v.name is None:
#             #if is_scan_op(v):
#             #    return 'scan'
#             global var_counter
#             var_counter += 1
#             v.name = 'v'+str(var_counter)
#             return 'v'+str(var_counter)
#         return save_name(v.name)
#     return '_'.join([save_name(p.name) for p in v.path])

def format_node(v):
    try:
        form = build_formula(v,True,20)
    except:
        form = ''
        pass
    if has_convis_attribute(v,'html_name') and get_convis_attribute(v,'html_name') is not None:
        if len(form) < 60 and form != '':
            return full_path(v)+" [label=<"+get_convis_attribute(v,'html_name')+" = "+form+">];"
        return full_path(v)+" [label=<"+get_convis_attribute(v,'html_name')+">];"
    if has_convis_attribute(v,'name') and get_convis_attribute(v,'name') is not None:
        if len(form) < 60 and form != '':
            return full_path(v)+" [label=\""+save_name(get_convis_attribute(v,'name'))+" = "+form+"\"];"
        return full_path(v)+" [label=\""+save_name(get_convis_attribute(v,'name'))+"\"];"
    return full_path(v)+";"

def explore_to_edge_of_node_iter(apply_node,depth=None,my_node=None,ignore=[], 
                        ignored_ops=[convis.theano.compile.ops.Shape,convis.theano.compile.ops.Shape_i],
                        do_scan_exploration=False):
    """ get variables that have a name """
    if my_node is None and hasattr(apply_node,'node'):
        my_node = apply_node.node
    node_counter = 0
    nodes_to_explore = [(apply_node,[])]
    nodes_explored = []
    scan_outputs = {}
    final_nodes_explored = []
    while len(nodes_to_explore) > 0:
        node,path = nodes_to_explore.pop()
        if node in nodes_explored:
            continue
        nodes_explored.append([node,path])
        if is_named_var(node) or is_scan_op(node):
            final_nodes_explored.append([node,path])
        if hasattr(node,'owner') and node.owner is not None and len(node.owner.inputs) > 0 and type(node.owner.op) not in ignored_ops:
            if is_named_var(node) or is_scan_op(node):
                new_path = path+[node]
            else:
                new_path = path
                #node_counter += 1
            if node in scan_outputs.keys():
                nodes_to_explore.append((scan_outputs[node],new_path))
                del scan_outputs[node]
            if has_convis_attribute(node,'copied_from'):
                nodes_to_explore.append((get_convis_attribute(node,'copied_from'),new_path))
            if do_scan_exploration:
                if not is_scan_op(node):
                    nodes_to_explore.extend([(i,new_path) for i in node.owner.inputs 
                                                     if not i in nodes_explored 
                                                 and 
                                                     not i in ignore])
                if is_scan_op(node):
                    ##nodes_to_explore.append((node.owner.op.graph.graph,new_path))
                    #print 'SCAN --- ',new_path, node.owner.op.outputs
                    nodes_to_explore.extend([(i,new_path) for i in node.owner.op.outputs 
                                                     if not i in nodes_explored 
                                                 and 
                                                     not i in ignore])
                    
                    scan_outputs.update(dict(zip(node.owner.op.inputs, node.owner.inputs[node.owner.op.n_seqs:])))
                    #scan_outputs.update(dict(zip(node.owner.inputs[node.owner.op.n_seqs:], node.owner.op.inputs)))
                    #print scan_outputs
                    if False:
                        nodes_to_explore.extend([(i,new_path) for i in node.owner.inputs 
                                                     if not i in nodes_explored 
                                                 and 
                                                     not i in ignore
                                                 and 
                                                     not i in convis.theano_utils.get_input_variables_iter(node.owner.op.outputs,include_copies=True,ignore=node.owner.inputs)])
                    #scan_outputs.update(dict([(i,i.copied_from) for i in node.owner.inputs 
                    #                                 if not i in nodes_explored 
                    #                             and 
                    #                                 not i in ignore
                    #                             and
                    #                                 hasattr(i,'copied_from')]))
                    #print scan_outputs
            else:
                nodes_to_explore.extend([(i,new_path) for i in node.owner.inputs 
                                                     if not i in nodes_explored 
                                                 and 
                                                     not i in ignore])
        if depth is not None and len(nodes_explored) > depth:
            break
    #print 'remaining:',scan_outputs
    return final_nodes_explored

def explore_to_all_nodes_iter(apply_node,depth=None,my_node=None,ignore=[], all_nodes=True, ignored_ops=[convis.theano.compile.ops.Shape,convis.theano.compile.ops.Shape_i]):
    """ get variables that have a name """
    if my_node is None and hasattr(apply_node,'node'):
        my_node = apply_node.node
    node_counter = 0
    nodes_to_explore = [(apply_node,[])]
    nodes_explored = []
    final_nodes_explored = []
    while len(nodes_to_explore) > 0:
        node,path = nodes_to_explore.pop()
        nodes_explored.append([node,path])
        if all_nodes or (is_var(node) or is_scan_op(node)):
            final_nodes_explored.append([node,path])
        if hasattr(node,'owner') and node.owner is not None and len(node.owner.inputs) > 0 and type(node.owner.op) not in ignored_ops:
            if all_nodes or (is_var(node) or is_scan_op(node)):
                new_path = path+[node]
            else:
                new_path = path
                #node_counter += 1
            if not is_scan_op(node):
                nodes_to_explore.extend([(i,new_path) for i in node.owner.inputs 
                                                 if not i in nodes_explored 
                                             and 
                                                 not i in ignore])
            if is_scan_op(node):
                nodes_to_explore.append((node.owner.op.graph.graph,new_path))
                #nodes_to_explore.extend([(i,new_path) for i in node.owner.op.outputs 
                #                                 if not i in nodes_explored 
                #                             and 
                #                                 not i in ignore])
        if depth is not None and len(nodes_explored) > depth:
            break
    return final_nodes_explored

options = {
    'variable_style': 'shape=box,style=filled,color=black,fillcolor=white',
    'parameter_style': "shape=box,style=\"\",color=white",
    'state_style': "shape=rpromoter,style=filled,color=black,fillcolor=yellow",
    'out_state_style': "shape=lpromoter,color=black,fillcolor=yellow",
    'output_style': "shape=rarrow,style=\"filled\",color=black,fillcolor=orange",
    'undefined_style': "shape=ellipse,style=\"\",color=black;label=\"\"",
}

dot = ""

dot += """digraph G {
    nodesep=.05;
    rankdir=LR;
    splines=ortho;
"""
# global inputs
dot += "    global_input [shape=invhouse, label=<<b>Input</b>>];\n"
# global outputs
#dot += "    global_output [shape=house,style=filled,color=orange, label=<<b>Output</b>>];\n"

all_dict = convis.base.create_hierarchical_dict_with_nodes(all_vars)
connections = []

def recursive_subgraphing(d,depth=0):
    global dot
    global connections
    for (subgraph,n) in d.iteritems():
        subgraph_name = save_name(subgraph.name)
        dot += "    subgraph cluster_"+subgraph_name+" {\n"
        if depth%2 == 0:
            dot += "        style=filled;\n        color=lightgray;\n"
        else:
            dot += "        style=filled;\n        color=lightblue;\n"
        if hasattr(subgraph,'html_name'):
            dot += "        label = <"+subgraph.html_name+">;\n"
        else:
            dot += "        label = \""+subgraph_name+"\";\n"
        boxes = {'input':[],'output':[],'state':[],'parameter':[],'variable':[]}
        for k,v in n.items():
            if has_convis_attribute(v,'variable_type') and get_convis_attribute(v,'variable_type') in boxes.keys():# and not hasattr(v,'copied_from'):
                boxes[get_convis_attribute(v,'variable_type')].append(v)
            else:
                if type(v) == dict:
                    recursive_subgraphing({k: v},depth=depth+1)
                else:
                    boxes['variable'].append(v)
        #dot += "        node [shape=rarrow,style=filled,color=black,fillcolor=green];\n" # input
        #dot += "        {rank=source; "+subgraph_name+"_input; }\n"
        dot += "        node ["+options['variable_style']+"];\n" # variables
        for v in boxes['variable']:
            dot += "        "+format_node(v)+"\n"
        dot += "        node ["+options['parameter_style']+"];\n" # parameters
        dot += "        { rank = min; \n"
        for v in boxes['parameter']:
            dot += "        "+format_node(v)+"\n"
        dot += "        }\n"
        dot += "        node ["+options['state_style']+"];\n" # states
        dot += "        { rank = same; \n"
        for v in boxes['state']:
            dot += "        "+format_node(v)+"\n"
        dot += "        }\n"
        dot += "        node [];\n" # out_states
        dot += "        { rank = same; \n"
        for k,v in n.items():
            if has_convis_attribute(v,'state_out_state'):
                dot += "        "+full_path(get_convis_attribute(v,'state_out_state'))+" ["+options['out_state_style']+"];\n"
        dot += "        }\n"
        dot += "        node ["+options['output_style']+"];\n" # outputs
        dot += "        { rank = same; \n"
        for v in boxes['output']:
            dot += "        "+format_node(v)+"\n"
        dot += "        }\n"            
        dot += "        node ["+options['undefined_style']+"];\n" # outputs
        ## connections to output
        # (fake for now)
        if False:
            dot += "        "+subgraph_name+"_input -> "+subgraph_name+"_compute;\n"
            for v in boxes['variable']:
                dot += "        "+full_path(v)+" -> "+subgraph_name+"_compute;\n"
            for v in boxes['parameter']:
                dot += "        "+full_path(v)+" -> "+subgraph_name+"_compute;\n"
            for v in boxes['state']:
                #print v.state_out_state
                #dot += "        "+v.state_out_state+" -> "+full_path(v)+"\n"
                dot += "        "+full_path(v)+" -> "+subgraph_name+"_compute;\n"
                dot += "        "+subgraph_name+"_compute -> "+full_path(v)+";\n"
            dot += "        "+subgraph_name+"_compute -> "+subgraph_name+"_output;\n"
        else:
            pass
        ## the fist option also draws the output of a subgraph if it is not part of any outputs!
        #paths = explore_to_edge_of_node_iter(subgraph.graph,depth=20000)
        ## this second option only includes the nodes that are part of the global tree
        for k,v in n.items():
            if has_convis_attribute(v,'variable_type'):
                paths = explore_to_edge_of_node_iter(v,depth=20000)
                for node,p in paths:
                    if len(p) > 0:
                        for i in range(len(p)-1):
                            connections.append((p[i],p[i+1], ''))
                        connections.append((p[-1],node, ''))  
        for v in boxes['state']:
            if has_convis_attribute(v,'state_out_state'):
                paths = explore_to_edge_of_node_iter(get_convis_attribute(v,'state_out_state'),depth=1000)
                for node,p in paths:
                    if len(p) > 0:
                        for i in range(len(p)-1):
                            connections.append((p[i],p[i+1], ''))
                        connections.append((p[-1],node, ''))
                connections.append((v,get_convis_attribute(v,'state_out_state'), ''))

        dot += "    }\n"
recursive_subgraphing(all_dict)

if False:
    for v in retina.outputs:
        paths = explore_to_edge_of_node_iter(v,depth=1000)
        for node,p in paths:
            if len(p) > 0:
                for i in range(len(p)-1):
                    connections.append((p[i],p[i+1], ' [color=black]'))
                connections.append((p[-1],full_path(node), ' [color=black]'))
    for v in all_vars:
        if has_convis_attribute(v,'state_out_state'):
            paths = explore_to_edge_of_node_iter(get_convis_attribute(v,'state_out_state'),depth=1000)
            print v.state_out_state, full_path(get_convis_attribute(v,'state_out_state'))
            for node,p in paths:
                if len(p) > 0:
                    for i in range(len(p)-1):
                        connections.append((p[i],p[i+1], ' [color=black]'))
                    connections.append((p[-1],full_path(node), ' [color=black]'))
            connections.append((full_path(v),full_path(get_convis_attribute(v,'state_out_state')), ' [color=red,style=dashed,penwidth=2.0,constraint=false]'))
connections = [(c[0],c[1]) for c in connections]
connections = convis.unique_list(connections)
print len(connections)
for c in connections:
    fmt = {};
    if (has_convis_attribute(c[1],'node') 
        and has_convis_attribute(c[0],'node') 
        and get_convis_attribute(c[1],'node') != get_convis_attribute(c[0],'node') 
        and len(get_convis_attribute(c[0],'path')) == len(get_convis_attribute(c[1],'path'))
        and len(get_convis_attribute(c[0],'path')) <= 2):
        fmt['color'] = 'black';
        fmt['penwidth'] = '4.0';
        fmt['minlen'] = '5.0';
        fmt['constraint'] = 'true';
    if convis.base.is_input(c[1]):
        fmt['color'] = 'black';
        fmt['penwidth'] = '4.0';
        fmt['minlen'] = '3.0';
    if convis.base.is_parameter(c[1]):
        fmt['penwidth'] = '2.0';
        fmt['style'] = 'solid';
        fmt['minlen'] = '1.0';
        fmt['color'] = 'blue';
        fmt['constraint'] = 'false';
    if has_convis_attribute(c[0],'state_out_state') and convis.base.is_out_state(c[1]) and get_convis_attribute(c[0],'state_out_state') == c[1]:
        fmt['style'] = 'dashed';
        fmt['color'] = 'yellow';
        fmt['penwidth'] = '2.0';
        fmt['minlen'] = '1.0';
        #fmt['penwidth'] = '2.0';
        fmt['constraint'] = 'false';
    fmt_str = '['+(', '.join([k+'='+v for (k,v) in fmt.items()]))+']'
    dot += "    "+full_path(c[1])+" -> "+full_path(c[0])+" "+fmt_str+";\n"
    #dot += "    "+full_path(c[1])+" -> "+full_path(c[0])+" "+c[2]+";\n"
print connections[:20]

if False:
    # we now have banished all copied from the plot!
    for n in all_vars:
        fmt = {};
        fmt['style'] = 'solid';
        fmt['color'] = 'green';
        if getattr(n,'name','') is None:
            continue
        if has_convis_attribute(n,'copied_from'):
            p1 = full_path(get_convis_attribute(n,'copied_from'))
            p2 = full_path(n)
            fmt_str = '['+(', '.join([k+'='+v for (k,v) in fmt.items()]))+']'
            dot += "    "+p1+" -> "+p2+" "+fmt_str+";\n"

#for v in all_vars:
#    if hasattr(v,'connects'):
#        for c in v.connects:
#            dot += '       '+ save_name(c[1].name)+'_output -> '+ save_name(c[0].name)+'_input;\n'
#for v in all_vars:
#    if hasattr(v,'out_state'):
#        for c in v.connects:
#            dot += '       '+ save_name(c[1].name)+'_output -> '+ save_name(c[0].name)+'_input;\n'
            
# finding free outputs
#for v in all_vars:
#    if hasattr(v,'variable_type') and v.variable_type == 'output':
#        if not hasattr(v,'connects'):
#            dot += "    "+full_path(v)+" -> global_output;\n"
dot += "    node [shape=rarrow,style=filled,color=black, fillcolor=orange, label=<<b>output</b>>];";
dot += "    { rank=same; ";
for i,v in enumerate(retina.outputs):
    dot += "    "+full_path(v)+"_global_output [label=<<b>outputs["+str(i)+"]</b>>];\n"
dot += "    }";
for i,v in enumerate(retina.outputs):
    dot += "    "+full_path(v)+" -> "+full_path(v)+"_global_output [minlen=2];\n"

# finding free inputs
for v in all_vars:
    if has_convis_attribute(v,'variable_type') and get_convis_attribute(v,'variable_type') == 'input':
        dot += "    global_input -> "+full_path(v)+";\n"


dot += """
}
"""

print '-'*40
print dot[:1000]
i = 10
with open('/home/jacob/Projects/convis/dot_test_'+str(i)+'.dot','w') as f:
    f.write(dot)
import os
os.system('dot /home/jacob/Projects/convis/dot_test_'+str(i)+'.dot -Tpng -o /home/jacob/Projects/convis/dot_test_'+str(i)+'.png')
#os.system('neato /home/jacob/Projects/convis/dot_test_'+str(i)+'.dot -Tpng -o /home/jacob/Projects/convis/dot_test_'+str(i)+'_neato.png')
#os.system('patchwork /home/jacob/Projects/convis/dot_test_'+str(i)+'.dot -Tpng -o /home/jacob/Projects/convis/dot_test_'+str(i)+'_patchwork.png')
#os.system('osage /home/jacob/Projects/convis/dot_test_'+str(i)+'.dot -Tpng -o /home/jacob/Projects/convis/dot_test_'+str(i)+'_osage.png')
#os.system('twopi /home/jacob/Projects/convis/dot_test_'+str(i)+'.dot -Tpng -o /home/jacob/Projects/convis/dot_test_'+str(i)+'_twopi.png')