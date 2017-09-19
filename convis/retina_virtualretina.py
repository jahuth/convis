"""

This module is a compatibility layer between the Virtual Retina configurations and behaviour and the python implementation.

"""
import xml.etree.ElementTree as ET
import collections
import copy
from future.utils import iteritems as _iteritems

def dict_recursive_update(d, u):
    for k, v in _iteritems(u):
        if isinstance(v, collections.Mapping):
            r = dict_recursive_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

# This dict contains tag and attribute names used in virtual retina configuration files
valid_retina_tags = {
    'retina': ['temporal-step__sec','input-luminosity-range','pixels-per-degree'],
    'basic-microsaccade-generator': ['pixels-per-degree',
                                'temporal-step__sec',
                                'angular-noise__pi-radians',
                                'period-mean__sec',
                                'period-stdev__sec',
                                'amplitude-mean__deg',
                                'amplitude-stdev__deg',
                                'saccade-duration-mean__sec',
                                'saccade-duration-stdev__sec'],
    'outer-plexiform-layer': [],
    'linear-version': ['center-sigma__deg',
                                'surround-sigma__deg',
                                'center-tau__sec',
                                'center-n__uint',
                                'surround-tau__sec',
                                'opl-amplification',
                                'opl-relative-weight',
                                'leaky-heat-equation'],
    'undershoot': ['relative-weight','tau__sec'],
    'contrast-gain-control': ['opl-amplification__Hz',
                                'bipolar-inert-leaks__Hz',
                                'adaptation-sigma__deg',
                                'adaptation-tau__sec',
                                'adaptation-feedback-amplification__Hz'],
    'ganglion-layer': ['sign',
                                'transient-tau__sec',
                                'transient-relative-weight',
                                'bipolar-linear-threshold',
                                'value-at-linear-threshold__Hz',
                                'bipolar-amplification__Hz',
                                'sigma-pool__deg'],
    'spiking-channel': ['g-leak__Hz',
                                'sigma-V',
                                'refr-mean__sec',
                                'refr-stdev__sec',
                                'random-init'],
    'square-array': ['size-x__deg', 'size-y__deg', 'uniform-density__inv-deg'],
    'circular-array': ['fovea-density__inv-deg','diameter__deg']
}

#only deep copy from this!
_default_config = {
            'basic-microsaccade-generator' :{
                'enabled': False,
                'pixels-per-degree':200,
                'temporal-step__sec':0.005,
                'angular-noise__pi-radians':0.3,
                'period-mean__sec':0.2,
                'period-stdev__sec':0,
                'amplitude-mean__deg':0.5,
                'amplitude-stdev__deg':0.1,
                'saccade-duration-mean__sec':0.025,
                'saccade-duration-stdev__sec':0.005,
            },
            'retina': {
                'temporal-step__sec':0.01,
                'input-luminosity-range':255,
                'pixels-per-degree':5.0
            },
            'log-polar-scheme' : {
                'enabled': False,
                'fovea-radius__deg': 1.0,
                'scaling-factor-outside-fovea__inv-deg': 1.0
            },
            'outer-plexiform-layers': [
                {
                    'linear-version': {
                        'center-sigma__deg': 0.05,
                        'surround-sigma__deg': 0.15,
                        'center-tau__sec': 0.01,
                        'center-n__uint': 0,
                        'surround-tau__sec': 0.004,
                        'opl-amplification': 10,
                        'opl-relative-weight': 1,
                        'leaky-heat-equation': 1,
                        'undershoot': {
                            'enabled': True,
                            'relative-weight': 0.8,
                            'tau__sec': 0.1
                        }                    
                    }
                }
            ],
            'contrast-gain-control': {
                'opl-amplification__Hz': 50, # for linear OPL: ampOPL = relative_ampOPL / fatherRetina->input_luminosity_range ;
                'bipolar-inert-leaks__Hz': 50,
                'adaptation-sigma__deg': 0.2,
                'adaptation-tau__sec': 0.005,
                'adaptation-feedback-amplification__Hz': 0
            },
            'ganglion-layers': [
                {
                    'name': 'Parvocellular On',
                    'enabled': True,
                    'sign': 1,
                    'transient-tau__sec':0.02,
                    'transient-relative-weight':0.7,
                    'bipolar-linear-threshold':0,
                    'value-at-linear-threshold__Hz':37,
                    'bipolar-amplification__Hz':100,
                    'sigma-pool__deg': 0.0,
                    'spiking-channel': {
                        'g-leak__Hz': 50,
                        'sigma-V': 0.1,
                        'refr-mean__sec': 0.003,
                        'refr-stdev__sec': 0,
                        'random-init': 0,
                        'square-array': {
                            'size-x__deg': 4,
                            'size-y__deg': 4,
                            'uniform-density__inv-deg': 20
                        }
                    }
                },
                {
                    'name': 'Parvocellular Off',
                    'enabled': True,
                    'sign': -1,
                    'transient-tau__sec':0.02,
                    'transient-relative-weight':0.7,
                    'bipolar-linear-threshold':0,
                    'value-at-linear-threshold__Hz':37,
                    'bipolar-amplification__Hz':100,
                    'sigma-pool__deg': 0.0,
                    'spiking-channel': {
                        'g-leak__Hz': 50,
                        'sigma-V': 0.1,
                        'refr-mean__sec': 0.003,
                        'refr-stdev__sec': 0,
                        'random-init': 0,
                        'square-array': {
                            'size-x__deg': 4,
                            'size-y__deg': 4,
                            'uniform-density__inv-deg': 20
                        }
                    }
                },
                {
                    'name': 'Magnocellular On',
                    'enabled': False,
                    'sign': 1,
                    'transient-tau__sec':0.03,
                    'transient-relative-weight':1.0,
                    'bipolar-linear-threshold':0,
                    'value-at-linear-threshold__Hz':80,
                    'bipolar-amplification__Hz':400,
                    'sigma-pool__deg': 0.1,
                    'spiking-channel': {
                        'g-leak__Hz': 50,
                        'sigma-V': 0,
                        'refr-mean__sec': 0.003,
                        'refr-stdev__sec': 0,
                        'random-init': 1,
                        'circular-array': {
                            'fovea-density__inv-deg': 15.0
                        }
                    }
                },
                {
                    'name': 'Magnocellular Off',
                    'enabled': False,
                    'sign': -1,
                    'transient-tau__sec':0.03,
                    'transient-relative-weight':1.0,
                    'bipolar-linear-threshold':0,
                    'value-at-linear-threshold__Hz':80,
                    'bipolar-amplification__Hz':400,
                    'sigma-pool__deg': 0.1,
                    'spiking-channel': {
                        'g-leak__Hz': 50,
                        'sigma-V': 0,
                        'refr-mean__sec': 0.003,
                        'refr-stdev__sec': 0,
                        'random-init': 1,
                        'circular-array': {
                            'fovea-density__inv-deg': 15.0
                        }
                    }
                }
            ]
        }
_config_info = {
            'basic-microsaccade-generator' :{
                'enabled': {
                        'default':False, 
                        'range':[False,True],
                        'doc': "The microsaccade generator is not implemented in convis."
                        },
                'pixels-per-degree':{ 'default': 200 },
                'temporal-step__sec':{ 'default': 0.005},
                'angular-noise__pi-radians':{ 'default': 0.3},
                'period-mean__sec':{ 'default': 0.2},
                'period-stdev__sec':{ 'default': 0},
                'amplitude-mean__deg':{ 'default': 0.5},
                'amplitude-stdev__deg':{ 'default': 0.1},
                'saccade-duration-mean__sec':{ 'default': 0.025},
                'saccade-duration-stdev__sec':{ 'default': 0.005},
            },
            'retina': {
                'temporal-step__sec':{ 'default': 0.01 },
                'input-luminosity-range':{ 'default': 255 },
                'pixels-per-degree':{ 'default': 5.0 }
            },
            'log-polar-scheme' : {
                'enabled': { 'default': False},
                'fovea-radius__deg': { 'default': 1.0},
                'scaling-factor-outside-fovea__inv-deg': { 'default': 1.0}
            },
            'outer-plexiform-layers': [
                {
                    'linear-version': {
                        'center-sigma__deg': { 
                            'default': 0.05
                            },
                        'surround-sigma__deg': { 
                            'default': 0.15
                            },
                        'center-tau__sec': {
                            'default': 0.01
                            },
                        'center-n__uint': {
                            'default': 0
                            },
                        'surround-tau__sec': {
                            'default': 0.004
                            },
                        'opl-amplification': {
                            'default': 10
                            },
                        'opl-relative-weight': {
                            'default': 1
                            },
                        'leaky-heat-equation': {
                            'default': 1
                            },
                        'undershoot': {
                            'enabled': { 'default': True},
                            'relative-weight': { 'default': 0.8},
                            'tau__sec': { 'default': 0.1}
                        }                    
                    }
                }
            ],
            'contrast-gain-control': {
                'opl-amplification__Hz': { 'default': 50}, # for linear OPL: ampOPL = relative_ampOPL / fatherRetina->input_luminosity_range ;
                'bipolar-inert-leaks__Hz': { 'default': 50},
                'adaptation-sigma__deg': { 'default': 0.2},
                'adaptation-tau__sec': { 'default': 0.005},
                'adaptation-feedback-amplification__Hz': { 'default': 0}
            },
            'ganglion-layers': [
                {
                    'name': { 
                        'default': 'Cell Layer' },
                    'enabled': { 
                        'default': True},
                    'sign': { 
                        'default': {'On': 1, 'Off': -1}, 
                        'values': [-1,1] },
                    'transient-tau__sec':{ 
                        'default': {'Parvocellular': 0.02, 'Magnocellular': 0.03}, 
                        'range': ['log',0.0001,1.0]},
                    'transient-relative-weight':{ 
                        'default': {'Parvocellular': 0.7, 'Magnocellular': 1.0},
                        'range':[0.0,1.0]},
                    'bipolar-linear-threshold':{ 
                        'default': 0, 
                        'var_name':'i_0_g',
                        'range': [-1.0,1.0]},
                    'value-at-linear-threshold__Hz':{ 
                        'default': {'Parvocellular': 37.0, 'Magnocellular': 80.0}
                        },
                    'bipolar-amplification__Hz':{ 
                        'default': {'Parvocellular': 0, 'Magnocellular':400},
                        'range': [0,600]
                        },
                    'sigma-pool__deg': {
                        'default': {'Parvocellular': 0.0, 'Magnocellular': 0.1}
                        },
                    'spiking-channel': {
                        'g-leak__Hz': { 
                            'default': 50
                            },
                        'sigma-V': { 
                            'default': 0.1
                            },
                        'refr-mean__sec': { 
                            'default': 0.003
                            },
                        'refr-stdev__sec': { 
                            'default': 0
                            },
                        'random-init': { 
                            'default': 1
                            },
                        'square-array': {
                            'size-x__deg': { 
                                'default': 4
                                },
                            'size-y__deg': { 
                                'default': 4
                                },
                            'uniform-density__inv-deg': { 
                                'default': 20
                                }
                        }
                    }
                }
            ]
        }


class RetinaConfiguration:
    """
        A configuration object that writes an xml file for VirtualRetina.

        (When this is altered, silver.glue.RetinaConfiguration has to also be updated by hand)

        Does not currently care to parse an xml file, but can save/load in json instead.
        The defaults are equal to `human.parvo.xml`.

        Values can be changed either directly in the configuration dictionary, or with the `set` helperfunction::

            config = silver.glue.RetinaConfiguration()
            config.retina_config['retina']['input-luminosity-range'] = 200
            config.set('basic-microsaccade-generator.enabled') = True
            config.set('ganglion-layers.*.spiking-channel.sigma-V') = 0.5 # for all layers

    """
    def __init__(self,updates={}):
        self.default()
        self.retina_config = dict_recursive_update(self.retina_config,updates)
    def default(self):
        """
        Generates a default config::

            self.retina_config = 
        """ + str(_default_config)

        self.retina_config = copy.deepcopy(_default_config)
    def get(self,key,default=None):
        """
            Retrieves values from the configuration.

            conf.set("ganglion-layers.*.spiking-channel.sigma-V",None) # gets the value for all layers
            conf.set("ganglion-layers.0",{}) # gets the first layer
        """
        if key == 'pixels-per-degree':
            return self.retina_config.get('retina',{}).get('pixels-per-degree',default)
        def get(config,key,default):
            if key == '':
                return config
            key = key.split('.')
            if type(config) == dict:
                if key[0] == '*':
                    # this will probably fail most of the time because the trees afterward are not identical
                    return [get(config[c],'.'.join(key[1:]),default) for c in config.keys()]
                if key[0] in config:
                    return get(config[key[0]],'.'.join(key[1:]),default)
            elif type(config) in (list,tuple):
                if key[0] == '*':
                    return [get(c,'.'.join(key[1:]),default) for c in config]
                return get(config[int(key[0])],'.'.join(key[1:]),default)
            return default

        return get(self.retina_config,key,default)
    def set(self,key,value,layer_filter=None):
        """
            shortcuts for frequent configuration values

            Knows where to put:

                'pixels-per-degree', 'size__deg' (if x and y are equal), 'uniform-density__inv-deg'
                all attributes of linear-version
                all attributes of undershoot

            Understands dot notation::

                conf = silver.glue.RetinaConfiguration()
                conf.set("ganglion-layers.2.enabled",True)
                conf.set("ganglion-layers.*.spiking-channel.sigma-V",0.101) # changes the value for all layers

            But whole sub-tree dicitonaries can be set as well (they replace, not update)::

                conf.set('contrast-gain-control', {'opl-amplification__Hz': 50,
                                                    'bipolar-inert-leaks__Hz': 50,
                                                    'adaptation-sigma__deg': 0.2,
                                                    'adaptation-tau__sec': 0.005,
                                                    'adaptation-feedback-amplification__Hz': 0
                                                })

            New dictionary keys are created automatically, new list elements can be created like this::

                conf.set("ganglion-layers.=.enabled",True) # copies all values from the last element
                conf.set("ganglion-layers.=1.enabled",True) # copies all values from list element 1
                conf.set("ganglion-layers.+.enabled",True) # creates a new (empty) dictionary which is probably underspecified

                conf.set("ganglion-layers.+",{
                            'name': 'Parvocellular On',
                            'enabled': True,
                            'sign': 1,
                            'transient-tau__sec':0.02,
                            'transient-relative-weight':0.7,
                            'bipolar-linear-threshold':0,
                            'value-at-linear-threshold__Hz':37,
                            'bipolar-amplification__Hz':100,
                            'spiking-channel': {
                                'g-leak__Hz': 50,
                                'sigma-V': 0.1,
                                'refr-mean__sec': 0.003,
                                'refr-stdev__sec': 0,
                                'random-init': 0,
                                'square-array': {
                                    'size-x__deg': 4,
                                    'size-y__deg': 4,
                                    'uniform-density__inv-deg': 20
                                }
                            }
                        }) # ganglion cell layer creates a new dicitonary

        """
        if key == 'pixels-per-degree':
            self.retina_config['retina']['pixels-per-degree'] = value
            if 'basic-microsaccade-generator' in self.retina_config and 'pixels-per-degree' in self.retina_config['basic-microsaccade-generator']:
                self.retina_config['basic-microsaccade-generator']['pixels-per-degree'] = value
        elif key in valid_retina_tags['linear-version']:
            self.retina_config['outer-plexiform-layers'][0]['linear-version'][key] = value
        elif key in valid_retina_tags['undershoot']:
            self.retina_config['outer-plexiform-layers'][0]['linear-version']['undershoot'][key] = value
        elif key == 'size__deg':
            for l in self.retina_config.get('ganglion-layers',[]):
                if layer_filter is not None and not layer_filter in l['name']:
                    continue
                l['spiking-channel'] = l.get('spiking-channel',{})
                l['spiking-channel']['square-array'] = l['spiking-channel'].get('square-array',{})
                l['spiking-channel']['square-array']['enabled'] = True
                l['spiking-channel']['square-array']['size-x__deg'] = value
                l['spiking-channel']['square-array']['size-y__deg'] = value
        elif key == 'uniform-density__inv-deg':
            for l in self.retina_config.get('ganglion-layers',[]):
                if layer_filter is not None and not layer_filter in l['name']:
                    continue
                l['spiking-channel'] = l.get('spiking-channel',{})
                l['spiking-channel']['square-array'] = l['spiking-channel'].get('square-array',{})
                l['spiking-channel']['square-array']['enabled'] = True
                l['spiking-channel']['square-array']['uniform-density__inv-deg'] = value
        elif key == 'enabled':
            for l in self.retina_config.get('ganglion-layers',[]):
                if layer_filter is not None and not layer_filter in l['name']:
                    continue
                l['enabled'] = value
        else:
            # Shortcut dot notation
            def put(d, keys, item):
                if "." in keys:
                    key, rest = keys.split(".", 1)
                    if type(d) is list:
                        if key == "+":
                            d.append({})
                            put(d[-1], rest, item)
                        elif key == "=":
                            d.append(d[-1])
                            put(d[-1], rest, item)
                        elif key.startswith("="):
                            d.append(d[int(key[1:])]) # use the referenced element
                            put(d[-1], rest, item)
                        elif key == "*":
                            for i in range(len(d)):
                                put(d[i], rest, item)
                        else:
                            while int(key) >= len(d):
                                d.append({})
                            put(d[int(key)], rest, item)
                    else:
                        if key == "*":
                            for k in d.keys():
                                put(d[k], rest, item)
                        else:
                            if key not in d:
                                d[key] = {}
                            put(d[key], rest, item)
                else:
                    if type(d) is list:
                        if key == "+":
                            d.append({})
                            d[-1] = item
                        else:
                            while int(key) >= len(d):
                                d.append({})
                            d[int(key)] = item
                    else:
                        d[keys] = item
            put(self.retina_config,key,value)
    def read_json(self,filename):
        """
            Reads a full retina config json file.
        """
        self.retina_config = json.load(filename)
    def _read_xml(self,filename):
        def get_attributes(tag_name,parent):
            attribs = {}
            for k in parent.attrib.keys():
                if k in valid_retina_tags[tag_name]:
                    attribs[k] = parent.attrib[k]
            return attribs
        def extend_dict(tag_name,d,config):
            # if possible, extend the dictionary with the sub key of this element
            try:
                d[tag_name] = get_attributes(tag_name,config.find(tag_name))
                return True
            except:
                pass
            return False
        tree = ET.parse(filename)
        root = tree.getroot()
        new_config = {}
        retina = root.find('retina')
        new_config['retina'] = get_attributes('retina',retina)
        
        # these nodes have no children:
        extend_dict('contrast-gain-control',new_config,retina)
        extend_dict('basic-microsaccade-generator',new_config,retina)
        extend_dict('log-polar-scheme',new_config,retina)

        # these might have children:        
        new_config['outer-plexiform-layers'] = []
        new_config['ganglion-layers'] = []
        for opl in retina.findall('outer-plexiform-layer'):
            opl_config = get_attributes('outer-plexiform-layer',opl)
            linear_version_config = get_attributes('linear-version',opl.find('linear-version'))
            extend_dict('undershoot',linear_version_config,opl.find('linear-version'))
            opl_config['linear-version'] = linear_version_config
            new_config['outer-plexiform-layers'].append(opl_config)
        for gl in retina.findall('ganglion-layer'):
            gl_config = get_attributes('ganglion-layer',gl)
            if extend_dict('spiking-channel',gl_config,gl):
                extend_dict('square-array',gl_config['spiking-channel'],gl.find('spiking-channel'))
                extend_dict('circular-array',gl_config['spiking-channel'],gl.find('spiking-channel'))
                try:
                    units = gl.find('spiking-channel').find('all-units')
                    gl_config['spiking-channel']['units'] = []
                    if units is not None:
                        for u in units.findall('unit'):
                            # we assume that all units have complete information or we fail
                            gl_config['spiking-channel']['units'].append({
                                'x': u.attrib.get('x-offset__deg'),
                                'y': u.attrib.get('y-offset__deg'),
                                'id': u.attrib.get('mvaspike-id')
                                })
                except:
                    raise
            new_config['ganglion-layers'].append(gl_config)
        return new_config
    def read_xml(self,filename):
        self.retina_config = self._read_xml(filename)
    def update_with_xml(self,filename):
        self.retina_config = dict_recursive_update(self.retina_config,self._read_xml(filename))
    def write_json(self,filename):
        json.dump(self.retina_config,filename)
    def write_json(self,filename):
        """
            Writes a retina config json file.
        """
        json.dump(self.retina_config,filename)
    def write_xml(self,filename):
        """
            Writes a full retina config xml file.

            Attributes that are not understood by the original Virtual Retina are removed.
        """
        def add_element(tag_name,parent,config,config_is_parent_config=True):
            if parent is None:
                return
            if config_is_parent_config:
                config = config.get(tag_name,{'enabled':False})
            if 'enabled' not in config or config['enabled']:
                e = ET.SubElement(parent, tag_name)
                for k in config.keys():
                    if k in valid_retina_tags[tag_name]:
                        e.set(k,str(config[k]))
                return e
        self.tree = ET.Element('retina-description-file')
        add_element('basic-microsaccade-generator', self.tree, self.retina_config)
        retina = add_element('retina', self.tree, self.retina_config)
        add_element('log-polar-scheme', retina, self.retina_config)
        for layer_config in self.retina_config.get('outer-plexiform-layers',[]):
            opl_layer = add_element('outer-plexiform-layer', retina, layer_config,False)
            lin = add_element('linear-version', opl_layer, layer_config, True)
            undershoot = add_element('undershoot', lin, layer_config.get('linear-version',{}))
        add_element('contrast-gain-control', retina, self.retina_config)
        layer_start = 0
        for layer_config in self.retina_config.get('ganglion-layers',[]):
            ganglion_layer = add_element('ganglion-layer', retina, layer_config,False)
            spiking_channel = add_element('spiking-channel', ganglion_layer, layer_config)
            if spiking_channel is not None and 'units' in layer_config.get('spiking-channel',{}):
                all_units = ET.SubElement(spiking_channel, 'all-units')
                for i,u in enumerate(layer_config.get('spiking-channel',{}).get('units',[])):
                    unit = ET.SubElement(all_units, 'unit')
                    unit.set('x-offset__deg',str(u.get('x',0.0)))
                    unit.set('y-offset__deg',str(u.get('y',0.0)))
                    unit.set('mvaspike-id',str(u.get('id',i+layer_start)))
                layer_start += len(layer_config.get('units',[]))
            square_array = add_element('square-array', spiking_channel, layer_config.get('spiking-channel',{'enabled':False}))
            circular_array = add_element('circular-array', spiking_channel, layer_config.get('spiking-channel',{'enabled':False}))
        with open(filename,'w') as f:
            f.write(ET.tostring(self.tree))
    def copy(self):
        import copy
        return RetinaConfiguration(copy.copy(self.retina_config))

def default_config():
    return RetinaConfiguration()

def random_config():
    return RetinaConfiguration()

def deriche_filter_density_map(retina, sigma0 = 1.0, Nx = None, Ny = None):
    """
        Returns a map of how strongly a point is to be blurred.

        Relevant config options of retina::

            'log-polar-scheme' : {
                'enabled': True,
                'fovea-radius__deg': 1.0,
                'scaling-factor-outside-fovea__inv-deg': 1.0
            }

        or for a circular (constant) scheme::

            'log-polar-scheme' : {
                'enabled': False,
                'fovea-radius__deg': 1.0,
                'scaling-factor-outside-fovea__inv-deg': 1.0
            } 

        The output should be used with `retina_base.deriche_coefficients` to generate the coefficient maps for a Deriche filter.
    """
    import numpy as np
    Ny = Nx if Ny is None else Ny
    x, y = np.meshgrid(np.arange(Nx),np.arange(Ny))
    midx = midy = Nx/2.0
    ratiopix = retina.degree_to_pixel(1.0)
    r = np.sqrt((x-midx)**2 + (y-midy)**2) + 0.001
    density = np.ones_like(r)
    log_polar_config = retina.config.retina_config.get('log-polar-scheme',{})
    if log_polar_config.get('enabled',False):
        log_polar_K = log_polar_config.get('scaling-factor-outside-fovea__inv-deg', 1.0)
        log_polar_R0 = log_polar_config.get('fovea-radius__deg', 1.0)
        if log_polar_K is None or log_polar_K < 0.0:
            log_polar_K = 1.0/log_polar_R0
        density = r
        density[r>log_polar_R0] = log_polar_R0 + log(1+log_polar_K*(density[r>log_polar_R0]-log_polar_R0))/log_polar_K
    return density/(sigma0*ratiopix)