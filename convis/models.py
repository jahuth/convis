from .base import Model
from . import filters
from .filters.simple import K_3d_kernel_filter, K_5d_kernel_filter, Nonlinearity, Nonlinearity_5d, Delay
import numpy as np
from variables import as_parameter, as_variable

class LN3d(Model):
    def __init__(self,config={},**kwargs):
        """
            :math:`I_{Gang}(x,y,t) = K_{x,y,t} * N(eT * V_{Bip})`

            :math:`N(V) = \\frac{i^0_G}{1-\lambda(V-v^0_G)/i^0_G}` (if :math:`V < v^0_G`)
            :math:`N(V) = i^0_G + \lambda(V-v^0_G)` (if :math:`V > v^0_G`)

        """
        if hasattr(config,'_'):
            # if the configuration is an Ox dictionary, only use the dictionary
            config = config._
        self.name = config.get('name','LN')
        self.linear_filter = K_3d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1))}),name='L')
        self.non_linear_filter = Nonlinearity(config.get('nonlinear',{}),name='N')
        self.non_linear_filter.add_input(self.linear_filter)
        super(LN3d,self).__init__(config=config,**kwargs)
        self.add_output(self.non_linear_filter.graph)


class LN(Model):
    def __init__(self,config={},**kwargs):
        """
            :math:`I_{Gang}(x,y,t) = K_{x,y,t} * N(eT * V_{Bip})`

            :math:`N(V) = \\frac{i^0_G}{1-\lambda(V-v^0_G)/i^0_G}` (if :math:`V < v^0_G`)
            :math:`N(V) = i^0_G + \lambda(V-v^0_G)` (if :math:`V > v^0_G`)

        """
        if hasattr(config,'_'):
            # if the configuration is an Ox dictionary, only use the dictionary
            config = config._
        self.name = config.get('name','LN')
        self.linear_filter = K_5d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1,1,1))}),name='L')
        self.non_linear_filter = Nonlinearity_5d(config.get('nonlinear',{}),name='N')
        self.non_linear_filter.add_input(self.linear_filter)
        super(LN,self).__init__(config=config,**kwargs)
        self.add_output(self.non_linear_filter.graph)


class Real_LN_model_5d(Model):
    def __init__(self,config={},**kwargs):
        """
            A linear temp-spatial filter and a rectification.

            See E. Real et al. 2017
        """
        if hasattr(config,'_'):
            # if the configuration is an Ox dictionary, only use the dictionary
            config = config._
        self.name = config.get('name','LN')
        self.linear_filter = K_5d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1,1,1))}),name='L')
        super(Real_LN_model,self).__init__(config=config,**kwargs)
        self.add_output(self.linear_filter.graph.sum(axis=(0,2,3,4)).clip(0,100000))

class Real_LN_model(Model):
    def __init__(self,config={},**kwargs):
        """
            A linear temporal-spatial filter and a rectification

            See E. Real et al. 2017
        """
        self.name = config.get('name','LN')
        self.linear_filter = filters.simple.RF_3d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1))}),name='L')
        super(Real_LN_model,self).__init__(config=config,**kwargs)
        self.add_output(self.linear_filter.graph.sum(axis=(1,2)).clip(0,100000))

class Real_LNSN_model(Model):
    def __init__(self,config={},**kwargs):
        """
            A subunit LN-LN cascade model.
            All subunits share the same recetive field at different positions.

            See E. Real et al. 2017
        """
        self.name = config.get('name','LN')
        self.bcm_linear_filter = filters.simple.K_3d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1))}),name='BCM')
        self.gcm_linear_filter = filters.simple.RF_3d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1))}),name='GCM')
        self.gcm_linear_filter += self.bcm_linear_filter.graph.clip(0,100000)
        super(Real_LNSN_model,self).__init__(config=config,**kwargs)
        self.add_output(self.gcm_linear_filter.graph.sum(axis=(1,2)).clip(0,100000))

        
class Real_LNSNF_model(Model):
    def __init__(self,config={},**kwargs):
        """
            A subunit LN-LN cascade model with temporal feedback.
            All subunits share the same recetive field at different positions.

            See E. Real et al. 2017
        """
        self.name = config.get('name','LN')
        self.bcm_linear_filter = filters.simple.K_3d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1))}),name='BCM')
        self.gcm_linear_filter = filters.simple.RF_3d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1))}),name='GCM')
        self.gcm_linear_filter += self.bcm_linear_filter.graph.clip(0,100000)
        super(Real_LNSNF_model,self).__init__(config=config,**kwargs)
        gcm_output = self.gcm_linear_filter.graph.clip(0,100000)
        self.gcm_feedback_filter = filters.simple.RecursiveFilter1dExponential(
            config.get('feedback',{'tau': 0.001}),
            model = self,
            name='GCM_feedback')
        self.gcm_feedback_filter += gcm_output
        self.gcm_delay = Delay({'delay':100},name='GCM_delay')
        self.gcm_delay += self.gcm_feedback_filter
        feedback_strength = as_parameter(0.00001,name='feedback_strength',doc="Output - feedback_strength * feedback")
        self.add_output((gcm_output - feedback_strength*self.gcm_delay.graph).sum(axis=(1,2)))


class Real_LNFSNF_model(Model):
    def __init__(self,config={},**kwargs):
        """
            A subunit LN-LN cascade model with temporal feedback at the first and second stage.
            All subunits share the same recetive field at different positions.

            See E. Real et al. 2017
        """
        self.name = config.get('name','LN')
        super(Real_LNFSNF_model,self).__init__(config=config,**kwargs)
        self.bcm_linear_filter = filters.simple.K_3d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1))}),name='BCM')
        bcm_output = self.bcm_linear_filter.graph.clip(0,100000)
        self.bcm_feedback_filter = filters.simple.RecursiveFilter1dExponential(
            config.get('bcm_feedback',{'tau': 0.001}),
            model = self,
            name='BCM_feedback')
        self.bcm_feedback_filter.add_input(bcm_output)
        self.bcm_delay = filters.simple.RF_3d_kernel_filter({'kernel':np.ones((1,1,1))},name='BCM_delays')
        bcm_feedback_strength = as_parameter(0.001,name='BCM_feedback_strength',doc="Output - feedback_strength * feedback")
        self.bcm_delay += bcm_output - bcm_feedback_strength * self.bcm_feedback_filter
        self.gcm_linear_filter = filters.simple.RF_2d_kernel_filter(config.get('linear',{'kernel':np.ones((20,20))}),name='GCM')
        self.gcm_linear_filter += self.bcm_delay.graph
        gcm_output = self.gcm_linear_filter.graph.clip(0,100000)
        self.gcm_feedback_filter = filters.simple.RecursiveFilter1dExponential(
            config.get('gcm_feedback',{'tau': 0.001}),
            model = self,
            name='GCM_feedback')
        self.gcm_feedback_filter += gcm_output
        self.gcm_delay = Delay({'delay':100},name='GCM_delay')
        self.gcm_delay += self.gcm_feedback_filter
        gcm_feedback_strength = as_parameter(0.00001,name='GCM_feedback_strength',doc="Output - feedback_strength * feedback")
        self.add_output((gcm_output - gcm_feedback_strength * self.gcm_delay.graph).sum(axis=(1,2)))

class Real_LNFDSNF_model(Model):
    def __init__(self,config={},**kwargs):
        """
            LNFSNF + individual delays after bipolar cell modules.
            (actually the same as LNFSNF in this implementation?)

            See E. Real et al. 2017
        """
        self.name = config.get('name','LN')
        super(Real_LNFDSNF_model,self).__init__(config=config,**kwargs)
        self.bcm_linear_filter = filters.simple.K_3d_kernel_filter(config.get('linear',{'kernel':np.ones((1,1,1))}),name='BCM')
        bcm_output = self.bcm_linear_filter.graph.clip(0,100000)
        self.bcm_feedback_filter = filters.simple.RecursiveFilter1dExponential(
            config.get('bcm_feedback',{'tau': 0.001}),
            model = self,
            name='BCM_feedback')
        self.bcm_feedback_filter.add_input(bcm_output)
        self.bcm_delay = filters.simple.RF_3d_kernel_filter({'kernel':np.ones((1,1,1))},name='BCM_delays')
        self.bcm_delay += bcm_output - 0.001 * self.bcm_feedback_filter
        self.gcm_linear_filter = filters.simple.RF_2d_kernel_filter(config.get('linear',{'kernel':np.ones((20,20))}),name='GCM')
        self.gcm_linear_filter += self.bcm_delay.graph
        gcm_output = self.gcm_linear_filter.graph.clip(0,100000)
        self.gcm_feedback_filter = filters.simple.RecursiveFilter1dExponential(
            config.get('gcm_feedback',{'tau': 0.001}),
            model = self,
            name='GCM_feedback')
        self.gcm_feedback_filter += gcm_output
        self.gcm_delay = Delay({'delay':100},name='GCM_delay')
        self.gcm_delay += self.gcm_feedback_filter
        self.add_output((gcm_output - 0.00001 * self.gcm_delay.graph).sum(axis=(1,2))) 