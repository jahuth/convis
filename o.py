"""
    Friendly `O` objects
    ====================

"""

from variable_describe import save_name


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