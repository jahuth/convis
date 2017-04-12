from .base import M
from .filters.simple import K_3d_kernel_filter, K_5d_kernel_filter, Nonlinearity, Nonlinearity_5d
import numpy as np

class LN3d(M):
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


class LN(M):
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


class Real_LN_model(M):
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

class Real_LNSN_model(M):
    pass
    def __init__(self,config={},**kwargs):
        """
            An LN subunit model: smaller LN models are integrated spatially.

            See E. Real et al. 2017
        """
        pass
class Real_LNSNF_model(M):
    pass
    def __init__(self,config={},**kwargs):
        """
            LNSN + Gain Control Circuit that feeds the output temproally filtered into the spatial integration.

            See E. Real et al. 2017
        """
        pass

class Real_LNFSNF_model(M):
    pass
    def __init__(self,config={},**kwargs):
        """
            LNSNF + Feedback for each bipolar cell module subunit.

            See E. Real et al. 2017
        """
        pass

class Real_LNFDSNF_model(M):
    def __init__(self,config={},**kwargs):
        """
            LNFSNF + individual delays after bipolar cell modules.

            See E. Real et al. 2017
        """
        pass    