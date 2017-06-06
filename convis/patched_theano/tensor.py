"""
This module provides patched theano tensor functions 
that return a wrapped theano variable.
"""
import theano.tensor as _T
from functools import wraps as _wraps
import inspect as _inspect
import types as _types

_variable_factory = lambda x: x # will be set later

def _getdefaultifnone(o,n,d):
    r = getattr(o,n,d)
    if r is None:
        return d
    return r

_functions_returning_variables = []
t_content = dir(_T)
for c in t_content:
    if type(getattr(_T,c)) == type:
        continue
    if hasattr(getattr(_T,c), '__call__'):
        _functions_returning_variables.append(c)

def _create_patched_function(name):
    original_f = getattr(_T,name)
    def tmp_f(*args,**kwargs):
        #print original_f
        #return original_f(*args,**kwargs)
        return _variable_factory(original_f(*args,**kwargs))
    if _inspect.isfunction(original_f):
        s = _wraps(original_f)(tmp_f) # invoking decorator
        s.__doc__ = ("This function was patched to return a convis variable.\n"
                 "The original definition is:\n\n"+ name 
                 + str(_inspect.formatargspec(*_inspect.getargspec(original_f))) + "\n"
                 + _getdefaultifnone(s,"__doc__",""))
    else:
        # a class
        s = tmp_f
        s.__doc__ = _getdefaultifnone(original_f,"__doc__","")
        try:
            if hasattr(original_f,'__init__'):
                s.__doc__ = ("This function was patched to return a convis variable.\n"
                     "The original definition is:\n\n"+ name 
                     + str(_inspect.formatargspec(*_inspect.getargspec(original_f.__init__))) + "\n"
                     + _getdefaultifnone(s,"__doc__",""))
        except:
            pass
    s.__doc__ += "\n    Note: Patched to return a convis variable."
    return s

_new_globals = {}
_new_globals.update(_T.__dict__)
del _new_globals['__package__']
del _new_globals['__path__']
del _new_globals['__file__']
del _new_globals['__name__']
globals()['__doc__'] += _new_globals['__doc__']
del _new_globals['__doc__']
for name in _functions_returning_variables:
    _new_globals[name] = _create_patched_function(name)
globals().update(_new_globals)
__all__ = _new_globals.keys()
