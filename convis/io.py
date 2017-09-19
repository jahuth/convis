import json
import numpy as np
from future.utils import iteritems as _iteritems

def _var_to_json_safe(v):
    if hasattr(v,'get_value'):
        v = v.get_value()
    if type(v) is np.ndarray:
        return v.tolist()
    return v
def _json_safe_to_value(v):
    if type(v) is list:
        try:
            return np.array(v)
        except:
            return v
    return v

def save_dict_to_json(filename,d):
    """
        Saves a (flat) dictionary that can also contain numpy
        arrays to a json file.
    """
    with open(filename,'w') as fp:
        dat = [(p,_var_to_json_safe(param)) for (p,param) in _iteritems(d)]
        json.dump(dict(dat), fp)
def load_dict_from_json(filename):
    """
        Loads a (flat) dictionary from a json file and converts
        lists back into numpy arrays.
    """
    with open(filename,'r') as fp:
        dat = json.load(fp)
        assert(type(dat) == dict)
        dat = dict([(p,_json_safe_to_value(param)) for (p,param) in _iteritems(dat)])
    return dat