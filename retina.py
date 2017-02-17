"""

This module implements a spiking retina model in python and theano.

It is based on the VirutalRetina Simualtor [Wohrer 2008].


General Overview
-----------------

The formulas on which the classes are based are:

$$C(x,y,t) = G * T(wu,Tu) * E(n,t) * L (x,y,t)$$
$$S(x,y,t) = G * E * C(x,y,t)$$ 
$$I_{OLP}(x,y,t) = \lambda_{OPL}(C(x,y,t) - w_{OPL} S(x,y,t)_)$$ 
$$\\\\frac{dV_{Bip}}{dt} (x,y,t) = I_{OLP}(x,y,t) - g_{A}(x,y,t)dV_{Bip}(x,y,t)$$
$$g_{A}(x,y,t) = G * E * Q(V{Bip})(x,y,t)`with $Q(V{Bip}) = g_{A}^{0} + \lambda_{A}V^2_{Bip}$$
$$I_{Gang}(x,y,t) = G * N(eT * V_{Bip})$$

with :math:`N(V) = \\\\frac{i^0_G}{1-\lambda(V-v^0_G)/i^0_G}` (if :math:`V < v^0_G`)

with :math:`N(V) = i^0_G + \lambda(V-v^0_G)` (if  :math:`V > v^0_G`)


"""
from .retina_virtualretina import RetinaConfiguration, default_config, random_config
from . import retina_virtualretina
from .base import M, GraphWrapper

from .filters.retina import *


class Retina(M):
    def __init__(self,config=None,**kwargs):
        if hasattr(config,'_'):
            # if the configuration is an Ox dictionary, only use the dictionary
            config = config._
        pixel_per_degree = None
        steps_per_second = None
        input_luminosity_range = None
        try:
            pixel_per_degree = float(config.get('retina',{}).get('pixels-per-degree',None))
        except:
            pass
        try:
            steps_per_second = 1.0/float(config.get('retina',{}).get('temporal-step__sec',None))
        except:
            pass
        try:
            input_luminosity_range = float(config.get('retina',{}).get('input-luminosity-range',None))
        except:
            pass
        super(Retina,self).__init__(pixel_per_degree=pixel_per_degree,steps_per_second=steps_per_second,input_luminosity_range=input_luminosity_range,**kwargs)
        self.config = config
        if self.config is None:
            self.config = RetinaConfiguration()
        self.input_luminosity_range = float(self.config.get('retina.input-luminosity-range',255.0))
        if kwargs.get('opl',True) == True:
            self.opl = OPLLayerNode(name='OPL',model=self,config=self.config.retina_config['outer-plexiform-layers'][0]['linear-version'])
        elif kwargs.get('opl',True) != False:
            self.opl = kwargs.get('opl',OPLLayerNode)(name='OPL',model=self,config=self.config.retina_config['outer-plexiform-layers'][0]['linear-version'])
            # todo: accept all arguments for layers
        if kwargs.get('bipolar',True):
            self.bipol = BipolarLayerNode(name='Bipolar',model=self,config=self.config.retina_config['contrast-gain-control'])
        self.ganglion_input_layers = []
        self.ganglion_spiking_layers = []
        for ganglion_config in self.config.retina_config.get('ganglion-layers',[]):
            if ganglion_config.get('enabled',True):
                gl_name = ganglion_config.get('name','')
                if gl_name != '':
                    gl_name = '_'+gl_name
                if kwargs.get('ganglion_input',True):
                    gang_in = GanglionInputLayerNode(name='GanglionInputLayer'+gl_name,model=self,config=ganglion_config)
                    self.ganglion_input_layers.append(gang_in)
                if kwargs.get('ganglion_spikes',True):
                    if 'spiking-channel' in ganglion_config and ganglion_config['spiking-channel'].get('enabled',True) != False:
                        gang_spikes = GanglionSpikingLayerNode(name='GanglionSpikes_'+gl_name,model=self,config=ganglion_config['spiking-channel'])
                        self.outputs.append(gang_spikes.output)
                        if kwargs.get('ganglion_input',True) and kwargs.get('ganglion_spikes',True):
                            #self.in_out(gang_in,gang_spikes)
                            gang_spikes.add_input(gang_in)
                        self.ganglion_spiking_layers.append(gang_spikes)
                if kwargs.get('bipolar',True) and kwargs.get('ganglion_input',True):
                    #self.in_out(self.bipol,gang_in)
                    gang_in.add_input(self.bipol)
        if kwargs.get('opl',True) and kwargs.get('bipolar',True):
            #self.in_out(self.opl,self.bipol)
            self.bipol.add_input(self.opl)
        #self.all = GraphWrapper([self.opl.output, self.bipol.output,self.ganglion_input_layers[0].output,self.ganglion_spiking_layers[0].output],'all',m=self)