from .base import M
from .filters.simple import K_3d_kernel_filter, Nonlinearity

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
        self.linear_filter = K_3d_kernel_filter(config.get('linear',{}),name='L')
        self.non_linear_filter = Nonlinearity(config.get('nonlinear',{}),name='N')
        self.non_linear_filter.add_input(self.linear_filter)
        super(LN,self).__init__(config=config,**kwargs)
        self.add_output(self.non_linear_filter.graph)