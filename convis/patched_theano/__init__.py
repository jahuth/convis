import theano as _theano
_new_globals = {}
_new_globals.update(_theano.__dict__)
del _new_globals['tensor']
del _new_globals['__package__']
del _new_globals['__path__']
del _new_globals['__file__']
del _new_globals['__name__']
globals().update(_new_globals)

from . import tensor