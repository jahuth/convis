# -*- coding: utf-8 -*-
"""
This module provides the retina model.
"""
from __future__ import print_function
from .base import Layer, Model,Output
from .retina_virtualretina import RetinaConfiguration
from .filters import retina as rf

class Retina(Layer):
    """
        A retinal ganglion cell model comparable to VirtualRetina [Wohrer2009]_.


        .. [Wohrer2009] Wohrer, A., & Kornprobst, P. (2009).
            Virtual Retina: a biological retina model and simulator, with contrast gain control.
            Journal of Computational Neuroscience, 26(2), 219-49. http://doi.org/10.1007/s10827-008-0108-4



        Attributes
        ----------

        opl : Layer (convis.filters.retina.OPL)
        bipolar : Layer (convis.filters.retina.Bipolar)
        gang_0_input : Layer (convis.filters.retina.GanglionInput)
        gang_0_spikes : Layer (convis.filters.retina.GanglionSpiking)
        gang_1_input : Layer (convis.filters.retina.GanglionInput)
        gang_1_spikes : Layer (convis.filters.retina.GanglionSpiking)
        _timing : list of tuples
            timing information of the last run (last chunk)
            Each entry is a tuple of (function that was executed, 
            number of seconds it took to execute)
        keep_timing_info : bool
            whether to store all timing information in a list
        timing_info : list
            stores timing information of all runs if
            `keep_timing_info` is True.


        Examples
        --------


        .. plot::
            :include-source:
            
            import convis
            import numpy as np
            from matplotlib import pylab as plt
            retina = convis.retina.Retina()
            inp = convis.samples.moving_grating(2000)
            inp = np.concatenate([inp[:1000],2.0*inp[1000:1500],inp[1500:2000]],axis=0)
            o_init = retina.run(inp[:500],dt=200)
            o = retina.run(inp[500:],dt=200)
            convis.plot_5d_time(o[0],mean=(3,4)) # plots the mean activity over time
            plt.figure()
            retina = convis.retina.Retina(opl=True,bipolar=False,gang=True,spikes=False)
            o_init = retina.run(inp[:500],dt=200)
            o = retina.run(inp[500:],dt=200)
            convis.plot_5d_time(o[0],mean=(3,4)) # plots the mean activity over time
            convis.plot_5d_time(o[0],alpha=0.1) # plots a line for each pixel




        See Also
        --------

        convis.base.Layer : The Layer base class, providing chunking and optimization
        convis.filters.retina.OPL : The outer plexiform layer performs luminance to contrast conversion
        convis.filters.retina.Bipolar : provides contrast gain control
        convis.filters.retina.GanglionInput : provides a static non-linearity and a last spatial integration
        convis.filters.retina.GanglionSpiking : creates spikes from an input current


    """
    def __init__(self,opl=True,bipolar=True,gang=True,spikes=True):
        self.keep_timing_info = False
        self.timing_info = []
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
        conf = RetinaConfiguration()
        self.parse_config(conf)
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
        self._timing = []
        import datetime
        io_buffers = {'I1':inp}
        for b_out,f,b_in in self.commands:
            start_time = datetime.datetime.now()
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
            self._timing.append((f,(datetime.datetime.now()-start_time).total_seconds()))
        if self.keep_timing_info:
            self.timing_info.append(self._timing)
        return Output([io_buffers['I1'],io_buffers['I2']],keys=['ganglion_spikes_ON','ganglion_spikes_OFF'])
