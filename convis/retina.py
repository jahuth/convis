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
from __future__ import print_function
from .base import Layer, Model,Output
from .retina_virtualretina import RetinaConfiguration
from .filters import retina as rf

class Retina(Layer):
    def __init__(self,opl=True,bipolar=True,gang=True,spikes=True):
        super(Retina,self).__init__()
        self.opl = rf.OPL()
        self.bipolar = rf.Bipolar()
        self.gang_0_input = rf.GanglionInput()
        self.gang_0_spikes = rf.GanglionSpiking()
        self.gang_1_input = rf.GanglionInput()
        self.gang_1_spikes = rf.GanglionSpiking()
        def add(x,y):
            return x+y

        # Ix = f(Iy,Iz,...)
        self.commands = []
        if opl:
            self.commands.append((['I1'], self.opl, ['I1']))
        if bipolar:
            self.commands.append((['I1','I2'], self.bipolar, ['I1']))
        else:
            self.commands.append((['I1','I2'], 'copy', ['I1']))
        if gang:
            self.commands.append((['I1'], self.gang_0_input, ['I1']))
        if spikes:
            self.commands.append((['I1'], self.gang_0_spikes, ['I1']))
        if gang:
            self.commands.append((['I2'], self.gang_1_input, ['I2']))
        if spikes:
            self.commands.append((['I2'], self.gang_1_spikes, ['I2']))
    def cuda(self, *args, **kwargs):
        # for now, the modules are not collected!
        super(Retina,self).cuda(*args, **kwargs)
        self.opl.opl_filter.cuda(*args, **kwargs)
        self.opl.opl_filter.center_E.cuda(*args, **kwargs)
        self.opl.opl_filter.surround_E.cuda(*args, **kwargs)
        self.opl.opl_filter.center_G.cuda(*args, **kwargs)
        self.opl.opl_filter.surround_G.cuda(*args, **kwargs)
        self.opl.opl_filter.center_undershoot.cuda(*args, **kwargs)
        self.bipolar.cuda(*args, **kwargs)
        self.gang_0_input.cuda(*args, **kwargs)
        self.gang_0_spikes.cuda(*args, **kwargs)
        self.gang_1_input.cuda(*args, **kwargs)
        self.gang_1_spikes.cuda(*args, **kwargs)
    def parse_config(self,config,prefix='',key='retina_config_key'):
        if type(config) is str:
            config_file = config
            config = RetinaConfiguration()
            config.read_xml(config_file)
        self.opl.parse_config(config,prefix='outer-plexiform-layers.0.linear-version.',key=key)
        self.bipolar.parse_config(config,prefix='contrast-gain-control.',key=key)
        self.gang_0_input.parse_config(config,prefix='ganglion-layers.0.',key=key)
        self.gang_0_spikes.parse_config(config,prefix='ganglion-layers.0.spiking-channel.',key=key)
        self.gang_1_input.parse_config(config,prefix='ganglion-layers.1.',key=key)
        self.gang_1_spikes.parse_config(config,prefix='ganglion-layers.1.spiking-channel.',key=key)
    def forward(self,inp):
        io_buffers = {'I1':inp}
        for b_out,f,b_in in self.commands:
            if f =='copy':
                for oi,oo in enumerate(b_out):
                    io_buffers[oo] = io_buffers[b_in[0]]
            else:
                #print b_in, f, b_out
                o = f(*[io_buffers[i] for i in b_in])
                if type(o) is Output:
                    o = o[0] # we can only use the first output
                for oi,oo in enumerate(b_out):
                    io_buffers[oo] = o
        return Output([io_buffers['I1'],io_buffers['I2']],keys=['ganglion_spikes_ON','ganglion_spikes_OFF'])
