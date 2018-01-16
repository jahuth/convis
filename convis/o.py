"""
    Friendly `O` objects
    ====================

    `O` objects are python objects that can be initialized easily.
    They are used as user friendly dictionaries, as their attributes
    can be tab-completed in interactive python evnironments.

    To convert a dictionary into an `O`, supply it as keyword arguemnts::

        a_dictionary = {'something': "a string", 'x': 0, 'y': 2}
        an_o = O(**a_dictionary)
        print an_o.x

    Or supply it to an already created `O` object to create a new one:

        a_second_dictionary = {'x':5, 'z':7}
        a_new_o = an_o(**a_second_dictionary)

    Methods you would use on a dictionary are renamed with with surrounding double underscores
    to avoid name conflicts: All attributes/methods *without* a leading underscrore
    are user supplied!

        a_new_o.__iteritems__() # is the same as a_new.__dict__.iteritems() or .items() in Python 3

    The `Ox` object is an *extended* version of an `O` object. It converts
    lists, dictionaries and tuples *recursively*. Also it provides the special
    attributes `._all` and `._search` to find variables in a nested structure.


    The `O` objects included in this toolbox have additional behaviour to make it
    easier to work with theano parameters. Normally, assigning a value to an attribute
    is ignored, but for the case of `.parameter` objects, when a numeric object is
    assigned to a shared parameter, the value will be supplied to the parameters `set_value`
    function. If a theano variable is supplied or an object with a `get_graph` method,
    the shared parameter will be replaced everywhere it occurs with the subgraph.
    This can be used to switch between a parameterized and a dense kernel.

"""
from future.utils import iteritems as _iteritems

var_name_counter = 0
def save_name(n):
    """
        Makes a name save to be used as an attribute.

        When the object supplied has a `name` attribute,
        it is used instead. If this `name` attribute contains
        an `!`, only the letters after the last `!` will be used.
    """
    if type(n) != str:
        if getattr(n,'name',None) is not None:
            n = n.name
            if '!' in n:
                n = n.split('!')
                if len(n[-1]) > 0:
                    # something!name -> name
                    n = n[-1]
                elif len(n) > 1:
                    # something! -> something
                    n = n[-2]
                else:
                    # n.name == '!'?
                    n = '_'
        else:
            if type(n) in [int,float]:
                return str(n)
            if n is None:
                return 'None'
            try:
                global var_name_counter
                n.name = 'unnamed_variable_'+str(var_name_counter)
                n = n.name
                var_name_counter += 1
            except:
                raise Exception('save_name got a '+str(type(n))+' instead of a string or object with name attribute.')
    n = n.replace(' ', '_').replace('-', '_').replace('+', '_').replace('*', '_').replace('&', '_').replace('.', '_dot_').replace(':', '_').replace('<','lt').replace('>','gt').replace('=','eq').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('!', '_')
    if len(n) >= 1 and n[0] in '0123456789':
        n = 'n'+n
    return n

try:
    import cPickle as pickle
except:
    import pickle

def find_a_class(class_name):
    class_path = class_name.split('.')
    try:
        class_reference = globals()[class_path[0]]
    except:
        try:
            import importlib
            class_reference = importlib.import_module(class_path[0])
        except:
            class_reference = __import__(class_path[0])
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


        Whether entries can be modified can be controlled with the
        `._readonly` boolean flag which enables or disables direct
        attribute assignment.

        Two special cases always allow assignment:
    """
    _readonly = True
    def __init__(self,**kwargs):
        global oid
        self._id=oid
        oid += 1
        self.__dict__.update(**dict([(save_name(k),v) for (k,v) in kwargs.items()]))
    def __call__(self,**kwargs):
        self.__dict__.update(**dict([(save_name(k),v) for (k,v) in kwargs.items()]))
        return self
    def __repr__(self):
        if '_doc' in self.__dict__.keys():
            return str(self.__dict__.get('_doc','')) + '\n' +  'Choices: '+(', '.join(self.__iterkeys__()))
        return 'Choices: '+(', '.join(self.__iterkeys__()))
    def _repr_html_(self):
        if '_doc' in self.__dict__.keys():
            return str(self.__dict__.get('_doc','')) + '<br/>' +  'Choices: '+(', '.join(self.__iterkeys__()))
        return repr(self)
    def __len__(self):
        return len([k for k in self.__dict__.keys() if not k.startswith('_') or k is '_self'])
    def __iter__(self):
        return iter([v for (k,v) in self.__dict__.items() if not k.startswith('_') or k is '_self'])
    def __iterkeys__(self):
        return iter([k for (k,v) in self.__dict__.items() if not k.startswith('_') or k is '_self'])
    def __iteritems__(self):
        return iter([(k,v) for (k,v) in self.__dict__.items() if not k.startswith('_') or k is '_self']) 
    def __setattr__(self, name, value):
        if name in self.__dict__.keys():
            var_to_replace = getattr(self, name)
            import convis, theano
            if hasattr(var_to_replace,'set_value') and hasattr(value,'__array__'):
                #print 'has set value'
                var_to_replace.set_value(value.__array__())
            elif isinstance(value, convis.GraphWrapper) or isinstance(value, theano.Variable):
                # replace parameter with paramterized parameter if possible 
                ## only when getattr(self, name) is a parameter
                ## only when dimensions match
                # then
                ## we find the parameters children and replace it in each with the graph we get from value.get_graph(name=old_name)
                #### what about parameters that are used in multiple places?
                from variables import get_convis_attribute, set_convis_attribute
                
                if hasattr(var_to_replace,'_') and hasattr(var_to_replace._,'graph'):
                    # the variable is actually
                    var_to_replace = var_to_replace._.graph
                filter_node = get_convis_attribute(var_to_replace,'original_node',get_convis_attribute(var_to_replace,'node'))
                if hasattr(value,'_as_TensorVariable'):
                    value.name = name
                    value = value._as_TensorVariable()
                else:
                    set_convis_attribute(value,'name', name)
                set_convis_attribute(value,'original_node',filter_node)
                convis.theano_utils.replace(filter_node.output,var_to_replace,value)
                filter_node.label_variables(filter_node.output)
                return
        if self._readonly and not name.startswith('_'):
            return
        if name in self.__dict__.keys():
            #print 'setting in dictionary'
            self.__dict__[name] = value
        else:
            #print 'setting for object'
            object.__setattr__(self, name, value)
    def __setitem__(self,k,v):
        self.__dict__[save_name(k)] = v
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
        if hasattr(self,'_self'):
            return getattr(self,'_self')
        if hasattr(self,'_original'):
            return getattr(self,'_original')
        return self._get_as('_original_type',recursive=True)
    def _as_TensorVariable(self):
        if hasattr(self,'_original'):
            if hasattr(getattr(self,'_original'),'_as_TensorVariable()'):
                return getattr(self,'_original')._as_TensorVariable()
    def _get_as(self,as_type='_original_type',recursive=False):
        def get_as_if_has_get_as(v,v_type='_original_type',recursive=False):
            "if the obejct provides a `_get_as` method, we call it, otherwise it is a normal object"
            if hasattr(v,'_get_as'):
                return v._get_as(v_type,recursive=recursive)
            return v
        called_as_type = as_type
        if as_type == '_original_type':
            as_type = self.__dict__.get('_original_type', None)
        if as_type == 'meta_dict':
            return {'type': str(self.__dict__.get('_original_type', None)),
                    'data': dict([(k, get_as_if_has_get_as(v, called_as_type, True) if recursive else v) for (k,v) in self.__iteritems__()])
                   }
        if as_type == dict or as_type == 'dict':
            return dict([(k, get_as_if_has_get_as(v, called_as_type, True) if recursive else v) for (k,v) in self.__iteritems__()])
        if as_type == list or as_type == 'list':
            import re
            keys = filter(lambda x: re.match(r'^n\d+$',x) != None, self.__dict__.keys())
            sorted_keys = sorted(map(lambda x: (len(x),x), keys))
            l = [self[k[1]] for k in sorted_keys]
            return [get_as_if_has_get_as(i, called_as_type, True) if recursive else i for i in l]
        if type(as_type) is str:
            class_reference = find_a_class(as_type)
            if class_reference is not None:
                try:
                    if 'config' in class_reference.__init__.im_func.func_code.co_varnames:
                        return class_reference(config=self._get_as(dict,recursive=True))
                    else:
                        return class_reference(**self._get_as(dict,recursive=True))
                except:
                    # class_reference is not a python function
                    return class_reference(**self._get_as(dict,recursive=True))
        try:
            return as_type(**self._get_as(dict))
        except:
            pass
        raise Exception('type not recognized: '+str(as_type)+'')

class _Search(O):
    def __init__(self,**kwargs):
        self._things = kwargs
    def __getattr__(self,search_string):
        return O(**dict([(save_name(k),v) for (k,v) in self._things.items() if search_string in k]))
    def __repr__(self):
        return 'Choices: enter a search term, enter a dot and use autocomplete to see matching items.'

def numpy_to_ox(a):
    try:
        return Ox(_original_type='numpy.ndarray',shape=a.shape,buffer=a.tostring())
    except:
        return Ox(_original_type='numpy.array',object=a.tolist())

def _load_meta_dict(m):
    try:
        assert('type' in m.keys())
        assert('data' in m.keys())
        data = dict([(k,_load_meta_dict(v)) for (k,v) in m['data'].items()])
        return Ox(_original_type=m['type'],**data)
    except:
        return m

def load(filename):
    with open(filename,'r') as f:
        s = pickel.load(f)


class Ox(O):
    """
        An `Ox` object is an extended  O object that allows easy access to its members
        and automatically converts a hierarchical dictionaries and lists into a hierarchical `Ox` objects.

        The special attributes `._all` and `._search` provide access to a flattend dictionary, 
        each level in the path separated by an underscore.

        Names will be converted to save variable names. Spaces and special characters are replaced with '_'.
        Lists will be treated as dicitonaries with keys n0...nX according to the length of the list.
        Objects that have a `__as_Ox__` or `__as_dict__` method will also be converted into `Ox` objects.
        The original type is saved in teh attribute `_original_type` so that even a pickled version can be 
        restored if the same packages are available.

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
        for k,v in _iteritems(kwargs):
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
        if hasattr(thing,'__as_Ox__'):
            return thing.__as_Ox__()
        if type(thing) is dict:
            return Ox(**thing)
        if type(thing) is list:
            return Ox(_item=thing,_original_type='list',**dict([('n'+str(i),v) for i,v in enumerate(thing)]))
        if hasattr(thing,'as_ox_dict'):
            return Ox(_item=thing,_original_type=str(thing.type),**thing.as_ox_dict())
        if hasattr(thing,'__as_dict__'):
            return Ox(_item=thing,_original_type=str(thing.type),**thing.__as_dict__())
        return thing
    def __setattr__(self, name, value):
        if not name.startswith('_'):
            value = self.__make_Ox(value)
        super(Ox,self).__setattr__(name,value)
    def __getattribute__(self,key):
        if key.startswith('_'):
            return super(Ox,self).__getattribute__(key)
        return self.__make_Ox(self.__dict__[key])
    def _save(self,filename):
        with open(filename,'w') as f:
            pickle.dump(self._get_as('meta_dict',recursive=True),f)
    def _dumps(self):
        return pickle.dumps(self._get_as('meta_dict',recursive=True))
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
