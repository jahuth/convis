import litus
import numpy as np
import matplotlib.pylab as plt
import uuid
try:
    from exceptions import NotImplementedError
except ImportError:
    pass

from ..base import *
from ..filters import Conv1d, Conv2d, Conv3d, TIME_DIMENSION
from .. import variables
from ..filters.simple import TemporalLowPassFilterRecursive, TemporalHighPassFilterRecursive

"""
Todo:

 - [ ] transient filters
 - [ ] filter inits
 - [ ] transient filter for Ganglion Input
 - [ ] sign for Ganglion Input
 - [ ] Automatic padding!
 - [ ] parsing config


"""



class SeperatableOPLFilter(Layer):
    def __init__(self):
        super(SeperatableOPLFilter, self).__init__()
        self.dims = 5
        self.center_G = Conv3d(1, 1, (1,10,10))
        self.center_G.set_weight(1.0)
        self.center_G.gaussian(0.05)
        self.sigma_center = variables.VirtualParameter(self.center_G.gaussian,value=0.05,retina_config_key='center-sigma__deg').set_callback_arguments(resolution=variables.default_resolution)
        self.center_E = Conv3d(1, 1, (5,1,1))
        self.center_undershoot = Conv3d(1, 1, (5,1,1))
        self.center_E.weight.data[0,0,-5,0,0] = 1.0
        self.tau_center = variables.VirtualParameter(float,value=0.01,retina_config_key='center-tau__sec')
        self.n_center = variables.VirtualParameter(int,value=0,retina_config_key='center-n__uint')
        self.f_exp_center = variables.VirtualParameter(
            self.center_E.exponential).set_callback_arguments(tau=self.tau_center,n=self.n_center,resolution=variables.default_resolution)
        self.undershoot_tau_center = variables.VirtualParameter(
            float,
            value=0.1,
            retina_config_key='undershoot.tau__sec')
        self.undershoot_relative_weight_center = variables.VirtualParameter(
            float,
            value=0.8,
            retina_config_key='undershoot.relative-weight').set_callback_arguments(resolution=variables.default_resolution)
        self.f_undershoot = variables.VirtualParameter(
            self.center_undershoot.highpass_exponential).set_callback_arguments(
                tau=self.undershoot_tau_center,
                relative_weight=self.undershoot_relative_weight_center,
                resolution=variables.default_resolution)
        self.surround_G = Conv3d(1, 1, (1,19,19),padding=(0,9,9))
        self.surround_G.set_weight(1.0)
        self.surround_G.gaussian(0.15)
        self.sigma_surround = variables.VirtualParameter(self.surround_G.gaussian,value=0.05,retina_config_key='surround-sigma__deg').set_callback_arguments(resolution=variables.default_resolution)
        self.sigma_surround.set(0.05)
        self.surround_E = Conv3d(1, 1, (19,1,1),padding=(9,0,0))
        self.surround_E.bias.data[0] = 0.0
        self.surround_E.weight.data[0,0,2,0,0] = 1.0
        self.tau_surround = variables.VirtualParameter(self.surround_E.exponential,value=0.004,retina_config_key='surround-tau__sec').set_callback_arguments(even=True,adjust_padding=False,resolution=variables.default_resolution)
        self.input_state = State(torch.zeros((1,1,1,1,1)))
        self.relative_weight = Parameter(0.5,retina_config_key='opl-relative-weight')
    @property
    def filter_length(self):
        return int(self.center_E.weight.data.shape[TIME_DIMENSION] + self.center_undershoot.weight.data.shape[TIME_DIMENSION] - 2)
    @property
    def filter_width(self):
        return self.center_G.weight.data.shape[-2] - 1
    @property
    def filter_height(self):
        return self.center_G.weight.data.shape[-1] - 1
    @property
    def filter_padding_2d(self):
        return (self.filter_width/2,
                int(self.filter_width - self.filter_width/2),
                int(self.filter_height/2),
                int(self.filter_height - self.filter_height/2),
                0,0)
    def forward(self, x):
        if not (self.input_state.data.shape[TIME_DIMENSION] == 2*self.filter_length and
            self.input_state.data.shape[3] == x.data.shape[3] and
            self.input_state.data.shape[4] == x.data.shape[4]):
            self.input_state = torch.autograd.Variable(torch.zeros((x.data.shape[0],x.data.shape[1],2*self.filter_length,x.data.shape[3],x.data.shape[4])))
            x_init = x[:,:,:2*self.filter_length,:,:]
            self.input_state[:,:,(-x_init.data.shape[2]):,:,:] = x_init
            #torch.zeros((1,1,self.filter_length,x.data.shape[3],x.data.shape[4]))
        if self._use_cuda:
            self.input_state = self.input_state.cuda()
            x_pad = torch.cat([self.input_state.cuda(), x.cuda()], dim=TIME_DIMENSION)
        else:
            self.input_state = self.input_state.cpu()
            #print self.input_state, x
            x_pad = torch.cat([self.input_state.cpu(), x.cpu()], dim=TIME_DIMENSION)
        y = self.center_G(nn.functional.pad(x_pad,self.filter_padding_2d,'replicate'))
        y = self.center_E(y)
        y = self.center_undershoot(y)
        s = self.surround_G(y)
        s = self.surround_E(nn.functional.pad(y,((0,0,0,0,len(self.surround_E),0)),'replicate'))
        # torch.autograd.Variable(torch.f rom_numpy(np.array(1.0,dtype='float32'))) *

        y = y - self.relative_weight * s[:,:,-y.data.shape[TIME_DIMENSION]:,:,:]
        self.input_state = x_pad[:,:,-(2*self.filter_length):,:,:]
        #return y[:,:,(self.filter_length/2):-(self.filter_length/2),:,:]
        return y[:,:,self.filter_length:,:,:]

class HalfRecursiveOPLFilter(Layer):
    def __init__(self):
        super(HalfRecursiveOPLFilter, self).__init__()
        self.dims = 5
        self.center_G = Conv3d(1, 1, (1,10,10))
        self.center_G.set_weight(1.0)
        self.center_G.gaussian(0.05)
        self.sigma_center = variables.VirtualParameter(self.center_G.gaussian,value=0.05,retina_config_key='center-sigma__deg').set_callback_arguments(resolution=variables.default_resolution)
        self.center_E = TemporalLowPassFilterRecursive()
        self.tau_center = variables.VirtualParameter(float,value=0.01,retina_config_key='center-tau__sec',var=self.center_E.tau)
        self.n_center = variables.VirtualParameter(int,value=0,retina_config_key='center-n__uint')
        self.center_undershoot = TemporalHighPassFilterRecursive()
        self.undershoot_tau_center = variables.VirtualParameter(
            float,
            value=0.1,
            retina_config_key='undershoot.tau__sec',
            var=self.center_undershoot.tau)
        self.undershoot_relative_weight_center = variables.VirtualParameter(
            float,
            value=0.8,
            retina_config_key='undershoot.relative-weight',
            var=self.center_undershoot.k)
        self.surround_G = Conv3d(1, 1, (1,19,19), adjust_padding=False)
        self.surround_G.set_weight(1.0)
        self.surround_G.gaussian(0.15)
        self.sigma_surround = variables.VirtualParameter(self.surround_G.gaussian,value=0.05,retina_config_key='surround-sigma__deg').set_callback_arguments(resolution=variables.default_resolution)
        self.sigma_surround.set(0.05)
        self.surround_E = TemporalLowPassFilterRecursive()
        self.tau_surround = variables.VirtualParameter(float,value=0.004,retina_config_key='surround-tau__sec',var=self.surround_E.tau)
        self.input_state = State(torch.zeros((1,1,1,1,1)))
        self.relative_weight = Parameter(0.5,retina_config_key='opl-relative-weight')
        self.lambda_opl = Parameter(1.0,retina_config_key='opl-amplification')
    def clear(self):
        self.center_E.clear()
        self.center_undershoot.clear()
        self.surround_E.clear()
    @property
    def filter_width(self):
        return self.center_G.weight.data.shape[-2] - 1
    @property
    def filter_height(self):
        return self.center_G.weight.data.shape[-1] - 1
    @property
    def filter_padding_2d(self):
        return (int(self.filter_width/2),
                int(self.filter_width - self.filter_width/2),
                int(self.filter_height/2),
                int(self.filter_height - self.filter_height/2),
                0,0)
    def forward(self, x):
        if self._use_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        y = self.center_G(nn.functional.pad(x,self.filter_padding_2d,'replicate')) * self.lambda_opl
        y = self.center_E(y)
        y = self.center_undershoot(y)
        s = self.surround_G(nn.functional.pad(y,self.surround_G.kernel_padding,'replicate'))
        s = self.surround_E(s)
        self.center_signal = y
        self.surround_signal = s
        y = y - self.relative_weight * s[:,:,:,:,:]
        return y

class FullConvolutionOPLFilter(Layer):
    def __init__(self):
        super(FullConvolutionOPLFilter, self).__init__()
        self.dims = 5
        self.conv = nn.Conv3d(1, 1, (20,10,10))
        self.conv.bias.data[0] = 0.0
        self.conv.weight.data[:,:,:,:,:] = 0.0

    def forward(self, x):
        return self.conv(x)

    
class OPL(Layer):
    """
    The OPL current is a filtered version of the luminance input with spatial and temporal kernels.

    $$I_{OLP}(x,y,t) = \lambda_{OPL}(C(x,y,t) - w_{OPL} S(x,y,t)_)$$

    with:

    :math:`C(x,y,t) = G * T(wu,Tu) * E(n,t) * L (x,y,t)`

    :math:`S(x,y,t) = G * E * C(x,y,t)`

    In the case of leaky heat equation:

    :math:`C(x,y,t) = T(wu,Tu) * K(sigma_C,Tau_C) * L (x,y,t)`

    :math:`S(x,y,t) = K(sigma_S,Tau_S) * C(x,y,t)`
    p.275

    To keep all dimensions similar, a *fake kernel* has to be used on the center output that contains a single 1 but has the shape of the filters used on the surround, such that the surround can be subtracted from the center.

    The inputs of the function are: 

     * :py:obj:`L` (the luminance input), 
     * :py:obj:`E_n_C`, :py:obj:`TwuTu_C`, :py:obj:`G_C` (the center filters), 
     * :py:obj:`E_S`, :py:obj:`G_S` (the surround filters), 
     * :py:obj:`Reshape_C_S` (the fake filter), 
     * :py:obj:`lambda_OPL`, :py:obj:`w_OPL` (scaling and weight parameters)

    """
    def __init__(self,**kwargs):
        super(OPL, self).__init__()
        self.dims = 5
        self.opl_filter = HalfRecursiveOPLFilter()
    @property
    def filter_length(self):
        return self.opl_filter.filter_length
    def forward(self, x):
        return self.opl_filter(x)
 
class Bipolar(Layer):
    """
    Example Configuration::

        'contrast-gain-control': {
            'opl-amplification__Hz': 50, # for linear OPL: ampOPL = relative_ampOPL / fatherRetina->input_luminosity_range ;
                                                       # `ampInputCurrent` in virtual retina
            'bipolar-inert-leaks__Hz': 50,             # `gLeak` in virtual retina
            'adaptation-sigma__deg': 0.2,              # `sigmaSurround` in virtual retina
            'adaptation-tau__sec': 0.005,              # `tauSurround` in virtual retina
            'adaptation-feedback-amplification__Hz': 0 # `ampFeedback` in virtual retina
        },
    """
    def __init__(self,**kwargs):
        super(Bipolar, self).__init__()
        self.dims = 5
        self.lambda_amp = as_parameter(50.0,
                                        name="lambda_amp",
                                        retina_config_key='adaptation-feedback-amplification__Hz')
        self.g_leak = as_parameter(50.0,init=lambda x: float(x.node.config.get('bipolar-inert-leaks__Hz',50)),
                                    name="g_leak",
                                    retina_config_key='bipolar-inert-leaks__Hz')
        self.input_amp = as_parameter(50.0,
                                       init=lambda x: float(x.node.config.get('opl-amplification__Hz',100)),
                                       name="input_amp",
                                       retina_config_key='opl-amplification__Hz')
        self.inputNernst_inhibition = 0.0
                                    #as_parameter(0.0,init=lambda x: float(x.node.config.get('inhibition_nernst',0.0)),
                                    #               name="inputNernst_inhibition",
                                    #               retina_config_key='inhibition_nernst')
        self.tau = as_parameter(1.0,init=lambda x: x.resolution.seconds_to_steps(float(x.get_config_value('adaptation-tau__sec',0.00001))),
                           name = 'tau', retina_config_key='adaptation-tau__sec')
        self.steps = Variable(0.001)
        self.a_0 = Variable(1.0)
        self.a_1 = -(-self.steps/self.tau).exp()
        self.b_0 =  1.0 - self.a_1
        self.conv2d = Conv2d(1,1,(9,9),padding=(4,4))
        self.conv2d.gaussian(0.1)
        #self.conv2d.set_weight(1.0)
    def init_states(self,input_shapes):
        self.preceding_V_bip = Variable(torch.zeros((input_shapes[3],input_shapes[4])))
        self.preceding_attenuationMap = Variable(torch.ones((input_shapes[3],input_shapes[4])))
        self.preceding_inhibition = Variable(torch.ones((input_shapes[3],input_shapes[4])))
    def forward(self, x):
        y = self.y = torch.autograd.Variable(torch.zeros(x.data.shape))
        # initial slices
        if not hasattr(self,'preceding_V_bip'):
            self.init_states(x.data.shape)
        if self._use_cuda:
            #self.conv2d.cuda()
            x = x.cuda()
            y = y.cuda()
            self.a_1 = self.a_1.cuda()
            self.a_0 = self.a_0.cuda()
            self.b_0 = self.b_0.cuda()
            self.preceding_V_bip = self.preceding_V_bip.cuda()
            self.preceding_attenuationMap = self.preceding_attenuationMap.cuda()
            self.preceding_inhibition = self.preceding_inhibition.cuda()
        else:
            x = x.cpu()
            y = y.cpu()
            self.a_1 = self.a_1.cpu()
            self.a_0 = self.a_0.cpu()
            self.b_0 = self.b_0.cpu()
            self.preceding_V_bip = self.preceding_V_bip.cpu()
            self.preceding_attenuationMap = self.preceding_attenuationMap.cpu()
            self.preceding_inhibition = self.preceding_inhibition.cpu()
        preceding_V_bip = self.preceding_V_bip
        preceding_attenuationMap = self.preceding_attenuationMap
        preceding_inhibition = self.preceding_inhibition
        for i,input_image in enumerate(x.split(1,dim=2)):
            total_conductance = self.g_leak + preceding_inhibition
            attenuation_map = (-self.steps*total_conductance).exp()
            E_infinity = (self.input_amp * input_image[0,0,0,:,:] + self.inputNernst_inhibition * preceding_inhibition)/total_conductance
            V_bip = ((preceding_V_bip - E_infinity) * attenuation_map) + E_infinity # V_bip converges to E_infinity
            
            inhibition = (self.lambda_amp*(preceding_V_bip.unsqueeze(0).unsqueeze(0))**2.0 * self.b_0 
                                         - preceding_inhibition * self.a_1) / self.a_0
            # smoothing with mean padding
            inhibition = (self.conv2d(inhibition - inhibition.mean()) + inhibition.mean())[0,0,:,:]
            # next step and output
            preceding_V_bip, preceding_attenuationMap, preceding_inhibition = V_bip, attenuation_map, inhibition
            y[:,:,i,:,:] = V_bip
        self.preceding_V_bip = preceding_V_bip
        self.preceding_attenuationMap = preceding_attenuationMap
        self.preceding_inhibition = preceding_inhibition
        return y


class GanglionInput(Layer):
    """
    The input current to the ganglion cells is filtered through a gain function.

    :math:`I_{Gang}(x,y,t) = G * N(eT * V_{Bip})`


    :math:`N(V) = \\frac{i^0_G}{1-\lambda(V-v^0_G)/i^0_G}` (if :math:`V < v^0_G`)

    
    :math:`N(V) = i^0_G + \lambda(V-v^0_G)` (if :math:`V > v^0_G`)

        Example configuration:

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
                    ...
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
                    ...
                }
            },

    """
    def __init__(self,**kwargs):
        super(GanglionInput, self).__init__()
        self.dims = 5
        self.sign = Parameter(1.0, retina_config_key='sign')
        self.transient = Conv3d(1, 1, (5,1,1))
        self.transient_tau_center = variables.VirtualParameter(
            float,
            value=0.03,
            retina_config_key='transient-tau__sec')
        self.transient_relative_weight_center = variables.VirtualParameter(
            float,
            value=0.7,
            retina_config_key='transient-relative-weight').set_callback_arguments(resolution=variables.default_resolution)
        self.f_transient = variables.VirtualParameter(
            self.transient.highpass_exponential).set_callback_arguments(
                tau=self.transient_tau_center,
                relative_weight=self.transient_relative_weight_center,
                resolution=variables.default_resolution,
                adjust_padding=False)
        self.i_0 = Parameter(37.0, retina_config_key='value-at-linear-threshold__Hz')
        self.v_0 = Parameter(0.0, retina_config_key='bipolar-linear-threshold')
        self.lambda_G = Parameter(100.0, retina_config_key='bipolar-amplification__Hz')
        #self.high_pass = nn.Conv1d()
        self.input_state = State(torch.zeros((1,1,1,1,1)))
        self.spatial_pooling = Conv3d(1,1,(1,9,9))
        self.sigma_surround = variables.VirtualParameter(
            self.spatial_pooling.gaussian,
            value=0.0,
            retina_config_key='sigma-pool__deg').set_callback_arguments(
                adjust_padding=True,
                resolution=variables.default_resolution)
    @property
    def filter_length(self):
        return self.transient.weight.data.shape[TIME_DIMENSION] - 1 
    def forward(self, x):
        if not (self.input_state.data.shape[TIME_DIMENSION] == 2*self.filter_length and
            self.input_state.data.shape[3] == x.data.shape[3] and
            self.input_state.data.shape[4] == x.data.shape[4]):
            self.input_state = State(torch.zeros((x.data.shape[0],x.data.shape[1],2*self.filter_length,x.data.shape[3],x.data.shape[4])))
            if self._use_cuda:
                self.input_state = self.input_state.cuda()
        x_pad = torch.cat([self.input_state, x], dim=TIME_DIMENSION)
        x = self.sign * self.transient(x_pad)[:,:,self.filter_length:,:,:]
        n = (self.i_0/(1-self.lambda_G*(x-self.v_0)/self.i_0))
        n_greater = (self.i_0 + self.lambda_G*(x-self.v_0))
        cond = x > self.v_0
        n.masked_scatter_(cond,n_greater.masked_select(cond)) # pytorch docs should have an example about this
        self.input_state = x_pad[:,:,-2*self.filter_length:,:,:]
        return self.spatial_pooling(n)

class GanglionSpiking(Layer):
    """
    The ganglion cells recieve the gain controlled input and produce spikes. 

    When the cell is not refractory, :math:`V` moves as:

    $$ \\\\dfrac{ dV_n }{dt} = I_{Gang}(x_n,y_n,t) - g^L V_n(t) + \eta_v(t)$$

    Otherwise it is set to 0.
    """
    def __init__(self,**kwargs):
        super(GanglionSpiking, self).__init__()
        self.dims = 5
        # parameters
        self.refr_mu = Parameter(0.003,
                            retina_config_key='refr-mean__sec',
                            doc='The mean of the distribution of random refractory times (in seconds).')
        self.refr_sigma = Parameter(0.001,
                            retina_config_key='refr-stdev__sec',
                            doc='The standard deviation of the refractory time that is randomly drawn around `refr_mu`')
        self.noise_sigma = Parameter(0.1,
                            retina_config_key='sigma-V',
                            doc='Amount of noise added to the membrane potential.')
        self.g_L = Parameter(50.0,
                            retina_config_key='g-leak__Hz',
                            doc='Leak current (in Hz or dimensionless firing rate).')
        self.tau = Parameter(0.001,
                            retina_config_key='--should be inherited',
                            doc = 'Length of timesteps (ie. the steps_to_seconds(1.0) of the model.')
    def init_states(self,input_shape):
        self.zeros = torch.autograd.Variable(torch.zeros((input_shape[3],input_shape[4])))
        self.V = 0.5+0.2*torch.autograd.Variable(torch.rand((input_shape[3],input_shape[4]))) # membrane potential
        if self._use_cuda:
            self.refr = 1000.0*(self.refr_mu + self.refr_sigma *
                                torch.autograd.Variable(torch.randn((input_shape[3],input_shape[4]))).cuda())
        else:
            self.refr = 1000.0*(self.refr_mu + self.refr_sigma *
                                torch.autograd.Variable(torch.randn((input_shape[3],input_shape[4]))).cpu())
        self.noise_prev = torch.autograd.Variable(torch.zeros((input_shape[3],input_shape[4])))
    def forward(self, I_gang):
        g_infini = 50.0 # apparently?
        if not hasattr(self,'V'):
            self.init_states(I_gang.data.shape)
        if self._use_cuda:
            #y = torch.autograd.Variable(torch.zeros(I_gang.data.shape)).cuda()
            self.V = self.V.cuda()
            self.refr = self.refr.cuda()
            self.zeros = self.zeros.cuda()
            self.noise_prev = self.noise_prev.cuda()
        else:
            #y = torch.autograd.Variable(torch.zeros(I_gang.data.shape)).cpu()
            self.V = self.V.cpu()
            self.refr = self.refr.cpu()
            self.zeros = self.zeros.cpu()
            self.noise_prev = self.noise_prev.cpu()
        all_spikes = []
        for t, I in enumerate(I_gang.squeeze(0).squeeze(0)):
            #print I.data.shape, self.V.data.shape,torch.randn(I.data.shape).shape
            if self._use_cuda:
                noise = torch.autograd.Variable(torch.randn(I.data.shape)).cuda()
            else:
                noise = torch.autograd.Variable(torch.randn(I.data.shape)).cpu()
            V = self.V + (I - self.g_L * self.V + self.noise_sigma*noise*torch.sqrt(self.g_L/self.tau))*self.tau
            # canonical form: 
            #
            # V = V + (E_L - V + R*I)*dt/tau 
            #      + self.noise_sigma*noise*torch.sqrt(2.0*dt/tau)
            # with dt = self.tau
            #      tau = 1/self.g_L
            #      R = tau
            #      E_L = 0
            #
            if self._use_cuda:
                refr_noise = 1000.0*(self.refr_mu + self.refr_sigma *
                                    torch.autograd.Variable(torch.randn(I.data.shape)).cuda())
            else:
                refr_noise = 1000.0*(self.refr_mu + self.refr_sigma *
                                    torch.autograd.Variable(torch.randn(I.data.shape)).cpu())
            spikes = V > 1.0
            self.refr.masked_scatter_(spikes, refr_noise)
            self.refr.masked_scatter_(self.refr < 0.0, self.zeros)
            self.refr = self.refr - 1.0
            V.masked_scatter_(self.refr >= 0.5, self.zeros)
            V.masked_scatter_(spikes, self.zeros)
            self.V = V
            all_spikes.append(spikes[None,:,:])
        return torch.cat(all_spikes,dim=0)[None,None,:,:,:]
