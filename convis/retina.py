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
from .misc_utils import suppress
from .retina_virtualretina import RetinaConfiguration, default_config, random_config
from . import retina_virtualretina
from .base import Model, GraphWrapper

from .filters.retina import *


class Retina(Model):
    def __init__(self,config=None,**kwargs):
        """
            This class instantiates a model similar to the Virtual Retina simulator.

            It is comprised of an opl and a bipolar layer and N ganglion input and 
            ganglion spiking layers.

            The layers are connected like this::

                input -> opl -> bipol -> [ ganglion_input (eg On)  -> ganglion_spikes ] -> output[0] 
                                         [ ganglion_input (eg Off) -> ganglion_spikes ] -> output[1]

            Each layer can be disabled or overwritten by a different class by 
            providing the keyword arguments for this layer::

                Retina(config, opl=convis.filters.retina.OPLLayerLeakyHeatNode, bipolar=False, ganglion_input=SomeCustomClass)

            This example will disable the bipolar layer and replace opl and ganglion_input
            layers with other classes. By default, the outputs of one layer that should be
            fed into a layer that was disabled are added to the model outputs and layers
            that would have recieved input from a layer that was disabled expose their input.

            The changed model looks like this::

                input ->        opl (convis.filters.retina.OPLLayerLeakyHeatNode)        -> output[0]
                input -> [ ganglion_input (eg On, SomeCustomClass)  -> ganglion_spikes ] -> output[1] 
                input -> [ ganglion_input (eg Off, SomeCustomClass) -> ganglion_spikes ] -> output[2]

            This can be changed like this::

                retina = Retina(config, opl=convis.filters.retina.OPLLayerLeakyHeatNode, bipolar=False, ganglion_input=SomeCustomClass)
                for layer in retina.ganglion_input_layers:
                    # each ganglion_input recieves input from the opl
                    layer.add_input(retina.opl)
                # also we can remove the opl from the model output
                del retina.outputs[0]



            Note: Handeling the config is still a mess
        """
        self.name = 'Retina Model'
        if hasattr(config,'_'):
            # if the configuration is an Ox dictionary, only use the dictionary
            config = config._
        pixel_per_degree = None
        steps_per_second = None
        input_luminosity_range = None
        if debug in kwargs:
            self.debug = debug
        with suppress(Exception):
            pixel_per_degree = float(config.get('retina',{}).get('pixels-per-degree',None))
        with suppress(Exception):
            steps_per_second = 1.0/float(config.get('retina',{}).get('temporal-step__sec',None))
        with suppress(Exception):
            input_luminosity_range = float(config.get('retina',{}).get('input-luminosity-range',None))
        super(Retina,self).__init__(pixel_per_degree=pixel_per_degree,steps_per_second=steps_per_second,input_luminosity_range=input_luminosity_range,**kwargs)
        self.config = config
        if self.config is None:
            self.config = RetinaConfiguration()
        self.input_luminosity_range = float(self.config.get('retina.input-luminosity-range',255.0))


        def choose_class(key,default_class):
            if key == True:
                return default_class
            return kwargs.get(key,default_class)


        if kwargs.get('opl',True) :
            self.opl = choose_class('opl',OPLLayerNode)(name='OPL',model=self,config=self.config.retina_config['outer-plexiform-layers'][0]['linear-version'])
            if not kwargs.get('bipolar',True):
                self.add_output(self.opl)
                if self.debug:
                    print 'adding opl output'
        if kwargs.get('bipolar',True):
            self.bipol = choose_class('bipolar',BipolarLayerNode)(name='Bipolar',model=self,config=self.config.retina_config['contrast-gain-control'])
            if not kwargs.get('ganglion_input',True):
                self.add_output(self.bipol)
                if self.debug:
                    print 'adding bipolar output'
        self.ganglion_input_layers = []
        self.ganglion_spiking_layers = []
        for ganglion_config in self.config.retina_config.get('ganglion-layers',[]):
            if ganglion_config.get('enabled',True):
                gl_name = ganglion_config.get('name','')
                if gl_name != '':
                    gl_name = '_'+gl_name
                if kwargs.get('ganglion_input',True):
                    gang_in = choose_class('ganglion_input',GanglionInputLayerNode)(name='GanglionInputLayer'+gl_name,model=self,config=ganglion_config)
                    self.ganglion_input_layers.append(gang_in)
                    if not kwargs.get('ganglion_spikes',True):
                        self.add_output(gang_in)
                        if self.debug:
                            print 'adding ganglion input output'
                if kwargs.get('ganglion_spikes',True):
                    if 'spiking-channel' in ganglion_config and ganglion_config['spiking-channel'].get('enabled',True) != False:
                        gang_spikes = choose_class('ganglion_spikes',GanglionSpikingLayerNode)(name='GanglionSpikes_'+gl_name,model=self,config=ganglion_config['spiking-channel'])
                        self.outputs.append(gang_spikes.output)
                        if self.debug:
                            print 'adding ganglion spikes output'
                        if kwargs.get('ganglion_input',True):
                            gang_spikes.add_input(gang_in)
                            if self.debug:
                                print 'connecting ganglion input to ganglion spikes'
                        self.ganglion_spiking_layers.append(gang_spikes)

                if kwargs.get('bipolar',True) and kwargs.get('ganglion_input',True):
                    gang_in.add_input(self.bipol)
                    if self.debug:
                        print 'connecting bipolar to ganglion input'

        if kwargs.get('opl',True) and kwargs.get('bipolar',True):
            self.bipol.add_input(self.opl)
            if self.debug:
                print 'connecting opl and bipolar'
