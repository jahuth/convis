from ..variables import Parameter
from .. import variables
from ..base import Layer
import torch

_izhikevich_parameters = {
    'Tonic spiking':
        [0.02,0.2,-65.0,6.0,14.0],
    'Phasic spiking':   
        [0.02,0.25,-65.0,6.0,0.5],
    'Tonic bursting':        
        [0.02,0.2,-50.0,2.0,15.0],
    'Phasic bursting':        
        [0.02,0.25,-55.0,0.05,0.6],
    'Mixed mode':        
        [0.02,0.2,-55.0,4.0,10.0],
    'Sf. adaptation':        
        [0.01,0.2,-65.0,8.0,30.0],
    'Class 1':        
        [0.02,-0.1,-55.0,6.0,0.0],
    'Class 2':        
        [0.2,0.26,-65.0,0.0,0.0],
    'Spike latency':
        [0.02,0.2,-65.0,6.0,7.0],
    'Subthreshold osc':
        [0.05,0.26,-60.0,0.0,0.0],
    'Resonator':
        [0.1,0.26,-60.0,-1.0,0.0],
    'Integrator':
        [0.02,-0.1,-55.0,6.0,0.0],
    'Rebound spike':
        [0.03,0.25,-60.0,4.0,0.0],
    'Rebound burst':
        [0.03,0.25,-52.0,0.0,0.0],
    'Threshold var':
        [0.03,0.25,-60.0,4.0,0.0],
    'Bistability':
        [1.0,1.5,-60.0,0.0,-65.0],
    'DAP':
        [1.0,0.2,-60.0,-21.0,0.0],
    'Accomodation':
        [0.02,1.0,-55.0,4.0,0.0],
    'Inh-ind. spiking':
        [-0.02,-1.0,-60.0,8.0,80.0],
    'Inh-ind. bursting':
        [-0.026,-1.0,-45.0,0.0,80.0]
}

class Izhikevich(Layer):
    """Izhikevich Spiking Model with uniform parameters
    
    The Simple Model of Spiking Neurons after Eugene Izhikevich
    offers a wide range of neural dynamics with very few parameters.

    See: https://www.izhikevich.org/publications/spikes.htm

    Each pixel has two state variables: `v` and `u`.
    `v` corresponds roughly to the membrane potential of a neuron
    and `u` to a slow acting ion concentration. Both 
    variables influence each other dynamically:

        $$\\\\dot{v} = 0.04 \\\\cdot v^2 + 5 \\\\cdot v + 140 - u + I$$
        $$\\\\dot{u} = a \\\\cdot (b \\\\cdot v - u)$$

    If `v` crosses a threshold, it will be reset to a value `c` and
    `u` will be increased by another value `d`.

    The parameters of the model are:

        - `a`: relative speed between the evolution of `v` and `u`
        - `b`: amount of influence of `v` over `u`
        - `c`: the reset value for `v` if it crosses threshold
        - `d`: value to add to `u` if `v` crosses threshold

    
    Parameters
    ----------

    output_only_spikes (bool)
        whether only spikes should be returned (binary),
        or spikes, membrane potential and slow potential
        in one channel of the output each.

    Examples
    --------

    See also
    --------
    


    """
    def __init__(self,output_only_spikes=True,**kwargs):
        super(Izhikevich, self).__init__()
        self.dims = 5
        # parameters
        self.a = Parameter(0.02)
        self.b = Parameter(0.2)
        self.c = Parameter(-65.0, doc='reset value')
        self.d = Parameter(6.0, doc='increase of u when v is above threshold')
        self.threshold = Parameter(30.0)
        self.noise_strength = Parameter(0.001)
        self.register_state('v',None)
        self.register_state('u',None)
        self.iters = 2
        self.output_only_spikes = output_only_spikes
    def load_parameters_by_name(self, name=None):
        """Allows to load parameters for a range of behaviors.
        
        For a list of possible options, run the method without
        parameter or look at directly at the dictionary
        `convis.filters.spiking._izhikevich_parameters`.

        The dictionary has values for a,b,c,d and the recommended
        input.
        """
        if name in _izhikevich_parameters.keys():
            p = _izhikevich_parameters[name]
            self.a.set(p[0])
            self.b.set(p[1])
            self.c.set(p[2])
            self.d.set(p[3])
            print('Recommendet input: '+str(p[4]))
        else:
            print('Could not find key '+str(name))
            print('Posibilities are:\n'+'\n - '.join(_izhikevich_parameters.keys()))
    def init_states(self,input_shape):
        self.zeros = variables.zeros((input_shape[3],input_shape[4]))
        self.v = self.c*variables.ones((input_shape[3],input_shape[4])) # membrane potential
        self.u = self.b*self.c*variables.ones((input_shape[3],input_shape[4])) # slow potential
    def forward(self, I_in):
        if not hasattr(self,'v') or self.v is None:
            self.init_states(I_in.data.shape)
        all_spikes = []
        for t, I in enumerate(I_in.squeeze(0).squeeze(0)):
            noise = variables.randn(I.data.shape).cpu()
            noise_w = variables.randn(I.data.shape).cpu()
            dt = 1.0/float(self.iters)
            for i in range(self.iters):
                dv = dt*(0.04*self.v*self.v + 5.0* self.v + 140 - self.u + I)
                du = dt*self.a*(self.b*self.v - self.u)
                self.v = self.v + dv
                self.u = self.u + du
            spikes = self.v >= self.threshold
            self.v.masked_scatter_(spikes, self.zeros+self.c)
            self.u.masked_scatter_(spikes, self.u+self.d)
            spikes = spikes.float()
            if self.output_only_spikes:
                all_spikes.append(spikes[None,None,:,:])
            else:
                all_spikes.append(torch.cat([spikes[None,None,:,:],
                                             self.v[None,None,:,:],
                                             self.u[None,None,:,:]],dim=0))
        return torch.cat(all_spikes,dim=1)[None,:,:,:,:]

class RefractoryLeakyIntegrateAndFireNeuron(Layer):
    """LIF model with refractory period.

    Identical to `convis.filter.retina.GanglionSpiking`.

    The ganglion cells recieve the gain controlled input and produce spikes. 

    When the cell is not refractory, :math:`V` moves as:

    $$ \\\\dfrac{ dV_n }{dt} = I_{Gang}(x_n,y_n,t) - g^L V_n(t) + \eta_v(t)$$

    Otherwise it is set to 0.


    Attributes
    ----------

    refr_mu : Parameter
        The mean of the distribution of random refractory times (in seconds).
    refr_sigma : Parameter
        The standard deviation of the refractory time that is randomly drawn around `refr_mu`
    noise_sigma : Parameter
        Amount of noise added to the membrane potential.
    g_L : Parameter
        Leak current (in Hz or dimensionless firing rate).


    See Also
    --------

    convis.retina.Retina
    GanglionInput
    LeakyIntegrateAndFireNeuron
    """
    def __init__(self,**kwargs):
        super(RefractoryLeakyIntegrateAndFireNeuron, self).__init__()
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
        self.register_state('V',None)
        self.register_state('zeros',None)
        self.register_state('refr',None)
        self.register_state('noise_prev',None)
    def init_states(self,input_shape):
        self.zeros = variables.zeros((input_shape[3],input_shape[4]))
        self.V = 0.5+0.2*variables.rand((input_shape[3],input_shape[4])) # membrane potential
        if self._use_cuda:
            self.refr = 1000.0*(self.refr_mu + self.refr_sigma *
                                variables.randn((input_shape[3],input_shape[4])).cuda())
        else:
            self.refr = 1000.0*(self.refr_mu + self.refr_sigma *
                                variables.randn((input_shape[3],input_shape[4])).cpu())
        self.noise_prev = variables.zeros((input_shape[3],input_shape[4]))
    def forward(self, I_gang):
        g_infini = 50.0 # apparently?
        if not hasattr(self,'V') or self.V is None:
            self.init_states(I_gang.data.shape)
        if self._use_cuda:
            self.V = self.V.cuda()
            self.refr = self.refr.cuda()
            self.zeros = self.zeros.cuda()
            self.noise_prev = self.noise_prev.cuda()
        else:
            self.V = self.V.cpu()
            self.refr = self.refr.cpu()
            self.zeros = self.zeros.cpu()
            self.noise_prev = self.noise_prev.cpu()
        all_spikes = []
        for t, I in enumerate(I_gang.squeeze(0).squeeze(0)):
            #print I.data.shape, self.V.data.shape,torch.randn(I.data.shape).shape
            if self._use_cuda:
                noise = variables.randn(I.data.shape).cuda()
            else:
                noise = variables.randn(I.data.shape).cpu()
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
                                     variables.randn(I.data.shape).cuda())
            else:
                refr_noise = 1000.0*(self.refr_mu + self.refr_sigma *
                                     variables.randn(I.data.shape).cpu())
            spikes = V > 1.0
            self.refr.masked_scatter_(spikes, refr_noise)
            self.refr.masked_scatter_(self.refr < 0.0, self.zeros)
            self.refr = self.refr - 1.0
            V.masked_scatter_(self.refr >= 0.5, self.zeros)
            V.masked_scatter_(spikes, self.zeros)
            self.V = V
            all_spikes.append(spikes[None,:,:])
        return torch.cat(all_spikes,dim=0)[None,None,:,:,:]

class LeakyIntegrateAndFireNeuron(Layer):
    """LIF model.

    $$ \\\\dfrac{ dV_n }{dt} = I_{Gang}(x_n,y_n,t) - g^L V_n(t) + \eta_v(t)$$


    Attributes
    ----------

    noise_sigma : Parameter
        Amount of noise added to the membrane potential.
    g_L : Parameter
        Leak current (in Hz or dimensionless firing rate).


    See Also
    --------

    RefractoryLeakyIntegrateAndFireNeuron
    """
    def __init__(self,**kwargs):
        super(LeakyIntegrateAndFireNeuron, self).__init__()
        self.dims = 5
        # parameters
        self.noise_sigma = Parameter(0.1,
                            retina_config_key='sigma-V',
                            doc='Amount of noise added to the membrane potential.')
        self.g_L = Parameter(50.0,
                            retina_config_key='g-leak__Hz',
                            doc='Leak current (in Hz or dimensionless firing rate).')
        self.tau = Parameter(0.001,
                            retina_config_key='--should be inherited',
                            doc = 'Length of timesteps (ie. the steps_to_seconds(1.0) of the model.')
        self.register_state('V',None)
        self.register_state('zeros',None)
        self.register_state('noise_prev',None)
    def init_states(self,input_shape):
        self.zeros = variables.zeros((input_shape[3],input_shape[4]))
        self.V = 0.5+0.2*variables.rand((input_shape[3],input_shape[4])) # membrane potential
        self.noise_prev = variables.zeros((input_shape[3],input_shape[4]))
    def forward(self, I_gang):
        g_infini = 50.0 # apparently?
        if not hasattr(self,'V') or self.V is None:
            self.init_states(I_gang.data.shape)
        if self._use_cuda:
            self.V = self.V.cuda()
            self.zeros = self.zeros.cuda()
            self.noise_prev = self.noise_prev.cuda()
        else:
            self.V = self.V.cpu()
            self.zeros = self.zeros.cpu()
            self.noise_prev = self.noise_prev.cpu()
        all_spikes = []
        for t, I in enumerate(I_gang.squeeze(0).squeeze(0)):
            if self._use_cuda:
                noise = variables.randn(I.data.shape).cuda()
            else:
                noise = variables.randn(I.data.shape).cpu()
            V = self.V + (I - self.g_L * self.V + self.noise_sigma*noise*torch.sqrt(self.g_L/self.tau))*self.tau
            spikes = V > 1.0
            V.masked_scatter_(spikes, self.zeros)
            self.V = V
            all_spikes.append(spikes[None,:,:])
        return torch.cat(all_spikes,dim=0)[None,None,:,:,:]


class FitzHughNagumo(Layer):
    """Two state neural model.

    $$\\\\dot{v} = v - \\\\frac{1}{3} v^3 - w + I$$
    $$\\\\dot{w} = \\\\tau \\\\cdot (v - a - b \\\\cdot w) $$

    See also:

        - `Wikipedia on FitzHugh-Nagumo models <https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model>`_
    """
    def __init__(self,**kwargs):
        super(FitzHughNagumo, self).__init__()
        self.dims = 5
        self.a = Parameter(0.7)
        self.b = Parameter(0.8)
        self.tau = Parameter(0.08)
        self.noise_strength = Parameter(0.001)
        self.register_state('v',None)
        self.register_state('w',None)
        self.iters = 10
    def init_states(self,input_shape):
        self.v = 0.0*variables.randn((input_shape[3],input_shape[4])) # membrane potential
        self.w = 0.0*variables.randn((input_shape[3],input_shape[4])) # slow potential
    def forward(self, I_in):
        if not hasattr(self,'v') or self.v is None:
            self.init_states(I_in.data.shape)
        all_spikes = []
        for t, I in enumerate(I_in.squeeze(0).squeeze(0)):
            noise = variables.randn(I.data.shape).cpu()
            noise_w = variables.randn(I.data.shape).cpu()
            dt = 1.0/float(self.iters)
            for i in range(self.iters):
                dv = dt*(I + self.v - (self.v**3.0)/3.0 - self.w)
                dw = dt*(self.v - self.a -self.b*self.w)*self.tau
                self.v = self.v + dv + self.noise_strength * dt*noise
                self.w = self.w + dw + self.noise_strength * dt*noise_w
            all_spikes.append(self.v[None,:,:])
        return torch.cat(all_spikes,dim=0)[None,None,:,:,:]


class HogkinHuxley(Layer):
    """Neuron model of the giant squid axon.

    This model contains four state variables:
    the membrane potential `v` and three slow acting currents `n`, `m` and `h`.

    See also: 

        - `Wikipedia in Hodgkin-Huxley models <https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model>`_
        - `<http://neuronaldynamics.epfl.ch/online/Ch2.S2.html>`_

    """
    def __init__(self,**kwargs):
        super(HogkinHuxley, self).__init__()
        self.dims = 5
        # parameters
        self.gK = Parameter(36.0, doc='maximum conductance of Potassium channels')
        self.gNa = Parameter(120.0, doc='maximum conductance of Sodium channels')
        self.gL = Parameter(0.3, doc='leak current')
        self.Cm = Parameter(1.0, doc='membrane capacitance')
        self.VK = Parameter(-12.0,doc='potential of Potassium')
        self.VNa = Parameter(115.0,doc='potential of Sodium')
        self.Vl = Parameter(10.613,doc='potential of leak currents')
        self.noise_strength = Parameter(0.1)
        self.alpha_n = lambda v: (0.01 * (10.0 - v)) / (torch.exp(1.0 - (0.1 * v)) - 1.0)
        self.beta_n = lambda v : 0.125 * torch.exp(-v / 80.0)
        self.alpha_m = lambda v : (0.1 * (25.0 - v)) / (torch.exp(2.5 - (0.1 * v)) - 1.0)
        self.beta_m = lambda v : 4.0 * torch.exp(-v / 18.0)
        self.alpha_h = lambda v : 0.07 * torch.exp(-v / 20.0)
        self.beta_h = lambda v : 1.0 / (torch.exp(3.0 - (0.1 * v)) + 1.0)
        self.n_inf = lambda v : alpha_n(v) / (alpha_n(v) + beta_n(v))
        self.m_inf = lambda v : alpha_m(v) / (alpha_m(v) + beta_m(v))
        self.h_inf = lambda v : alpha_h(v) / (alpha_h(v) + beta_h(v))
        self.register_state('v',None)
        self.register_state('v_n',None)
        self.register_state('v_m',None)
        self.register_state('v_h',None)
        self.iters = 20
    def init_states(self,input_shape):
        self.v = 0.0*variables.randn((input_shape[3],input_shape[4])) # membrane potential
        self.v_n = 0.0*variables.randn((input_shape[3],input_shape[4])) # slow potential
        self.v_m = 0.0*variables.randn((input_shape[3],input_shape[4])) # slow potential
        self.v_h = 0.0*variables.randn((input_shape[3],input_shape[4])) # slow potential
    def forward(self, I_in):
        if not hasattr(self,'v') or self.v is None:
            self.init_states(I_in.data.shape)
        all_spikes = []
        for t, I in enumerate(I_in.squeeze(0).squeeze(0)):
            noise = variables.randn(I.data.shape).cpu()
            noise_w = variables.randn(I.data.shape).cpu()
            dt = 1.0/float(self.iters)
            for i in range(self.iters):
                dv = ((I + self.noise_strength*noise / self.Cm) 
                    - ((self.gK / self.Cm) * self.v_n**4.0 * (self.v - self.VK)) 
                    - ((self.gNa / self.Cm) * self.v_m**3.0 * self.v_h * (self.v - self.VNa)) 
                    - (self.gL / self.Cm * (self.v - self.Vl))
                )
                dn = (self.alpha_n(self.v) * (1.0 - self.v_n)) - (self.beta_n(self.v) * self.v_n)
                dm = (self.alpha_m(self.v) * (1.0 - self.v_m)) - (self.beta_m(self.v) * self.v_m)
                dh = (self.alpha_h(self.v) * (1.0 - self.v_h)) - (self.beta_h(self.v) * self.v_h)
                self.v = self.v + dt*dv 
                self.v_n = self.v_n + dt*dn
                self.v_m = self.v_m + dt*dm
                self.v_h = self.v_h + dt*dh
            all_spikes.append(self.v[None,:,:])
        return torch.cat(all_spikes,dim=0)[None,None,:,:,:]
