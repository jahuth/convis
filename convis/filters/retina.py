import litus
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pylab as plt
from ..theano_utils import conv3d, conv2d
import uuid
from exceptions import NotImplementedError

from ..base import *
from ..theano_utils import make_nd, dtensor5, pad5, pad5_txy, pad2_xy
from .. import retina_base
from ..numerical_filters import conv, fake_filter, fake_filter_shape
from ..numerical_filters import exponential_filter_1d, exponential_filter_5d, exponential_highpass_filter_1d, exponential_highpass_filter_5d
from ..numerical_filters import gauss_filter_2d, gauss_filter_5d

class OPLLayerNode(N):
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

    Since we want to have some temporal and some spatial convolutions (some 1d, some 2d, but orthogonal to each other), we have to use 3d convolution (we don't have to, but this way we never have to worry about which is which axis). 3d convolution uses 5-tensors (see: <a href="http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d">theano.tensor.nnet.conv</a>), so we define all inputs, kernels and outputs to be 5-tensors with the unused dimensions (color channels and batch/kernel number) set to be length 1.
    """
    def __init__(self,model=None,config={},name=None,input_variable=None):
        
        self.retina = model
        self.model = model
        self.set_config(config)

        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self.input_variable = make_nd(self.create_input(),5)
        self._E_n_C = self.shared_parameter(
            lambda x: exponential_filter_5d(tau=float(x.get_config('center-tau__sec',0.01,float)),
                                            n=int(x.get_config('center-n__uint', 0, int)),
                                            resolution=x.resolution),
                        name='E_n_C',
                        doc="The n-fold cascaded exponential creates a low-pass characteristic. A filter can be created with `numeric_filters.exponential_filter_5d`")
        self._TwuTu_C = self.shared_parameter(
            lambda x: exponential_highpass_filter_5d(tau = float(x.get_config('undershoot',{}).get('tau__sec',0.01)),
                                                     relative_weight=float(x.get_config('undershoot',{}).get('relative-weight', 0.8)),
                                                     resolution=x.resolution),name='TwuTu_C')
        self._G_C = self.shared_parameter(
            lambda x: gauss_filter_5d(float(x.get_config('center-sigma__deg',0.05)),float(x.get_config('center-sigma__deg',0.05)),
                                      resolution=x.resolution,even=False),name='G_C')
        self._E_S = self.shared_parameter(
            lambda x: exponential_filter_5d(tau=float(x.get_config('surround-tau__sec',0.004)),resolution=x.resolution),name='E_S')
        self._G_S = self.shared_parameter(
            lambda x: gauss_filter_5d(float(x.get_config('surround-sigma__deg',0.15)),float(x.get_config('surround-sigma__deg',0.15)),
                                      resolution=x.resolution,even=False),name='G_S')
        self._lambda_OPL = self.shared_parameter(
                lambda x: float(x.value_from_config()) / float(x.resolution.input_luminosity_range),
                save = lambda x: x.value_to_config(float(x.resolution.input_luminosity_range) * (float(x.var.get_value()))),
                get = lambda x: float(x.model.resolution.input_luminosity_range) * (float(x.var.get_value())),
                config_key = 'opl-amplification',
                config_default = 10.0,
                name='lambda_OPL',
                doc='Gain applied to the OPL signal.')
        self._w_OPL = self.shared_parameter(
            lambda x: x.get_config('opl-relative-weight',1.0,float),name='w_OPL')

        self._Reshape_C_S = self.shared_parameter(lambda x: fake_filter(get_convis_attribute(x.node._G_S,'update')(x).get_value(),
                                                                        get_convis_attribute(x.node._E_S,'update')(x).get_value()),name='Reshape_C_S',
                                                  doc='This filter resizes C such that the output has the same size as S.')

        self._input_init = as_state(dtensor5('input_init'),
                                    init=lambda x: np.zeros((1, get_convis_attribute(x.node._E_n_C,'update')(x).get_value().shape[1]-1+
                                                    get_convis_attribute(x.node._TwuTu_C,'update')(x).get_value().shape[1]-1+
                                                    get_convis_attribute(x.node._Reshape_C_S,'update')(x).get_value().shape[1]-1,
                                    1, x.input.shape[1], x.input.shape[2])))
        input_padded_in_time = T.concatenate([
                        self._input_init,
                        self.input_variable],axis=1)
        Nx = self._G_C.shape[3]-1 + self._G_S.shape[3]-1
        Ny = self._G_C.shape[4]-1 + self._G_S.shape[4]-1
        self._L = as_variable(pad5(pad5(input_padded_in_time,Nx,3),Ny,4),'L')
        self._C = GraphWrapper(as_variable(conv3d(conv3d(conv3d(self._L,self._E_n_C),self._TwuTu_C),self._G_C),'C'),name='center',ignore=[self._L]).graph
        self._S = GraphWrapper(as_variable(conv3d(conv3d(self._C,self._E_S),self._G_S),'S'),name='surround',ignore=[self._C]).graph
        I_OPL = as_variable(self._lambda_OPL * (conv3d(self._C,self._Reshape_C_S) - self._w_OPL * self._S),'I_OPL',html_name="I<sub>OPL</sub>",html_formula="I<sub>OPL</sub> = &lambda;*(C-w*S)")

        length_of_filters = self._E_n_C.shape[1]-1+self._TwuTu_C.shape[1]-1+self._Reshape_C_S.shape[1]-1 
        as_out_state(T.set_subtensor(self._input_init[:,-(input_padded_in_time[:,-(length_of_filters):,:,:,:].shape[1]):,:,:,:],
                                    input_padded_in_time[:,-(length_of_filters):,:,:,:]), self._input_init)
        super(OPLLayerNode,self).__init__(make_nd(I_OPL,3),name=name)

class OPLAllRecursive(N):
    """
    The OPL current is a filtered version of the luminance input with spatial and temporal kernels.

    The inputs of the function are: 

     * :py:obj:`L` (the luminance input), 
     * :py:obj:`lambda_OPL`, :py:obj:`w_OPL` (scaling and weight parameters)

    """
    def __init__(self,model=None,config={},name=None,input_variable=None):
        
        self.retina = model
        self.model = model
        self.set_config(config)
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self.input_variable = make_nd(self.create_input(),5)
        padding = (0,0,0)
        self._input_init = as_state(dtensor5('input_init'),
                                    init=lambda x: np.zeros((1, padding[0], 1, padding[1], padding[2])))
        input_padded_in_time = T.concatenate([
                        self._input_init,
                        self.input_variable],axis=1)
        Nx = 10#self._G_C.shape[3]-1 + self._G_S.shape[3]-1
        Ny = 10#self._G_C.shape[4]-1 + self._G_S.shape[4]-1
        self._L = pad5(pad5(input_padded_in_time,Nx,3),Ny,4)
        self._lambda_OPL = self.shared_parameter(
                lambda x: float(x.value_from_config()) / float(self.model.config.get('retina.input-luminosity-range',self.model.config.get('input-luminosity-range',255.0))),
                save = lambda x: x.value_to_config(float(self.model.config.get('retina.input-luminosity-range',self.model.config.get('input-luminosity-range',255.0))) * (float(x.var.get_value()))),
                get = lambda x: float(self.model.config.get('retina.input-luminosity-range',self.model.config.get('input-luminosity-range',255.0))) * (float(x.var.get_value())),
                config_key = 'opl-amplification',
                config_default = 10.0,
                name='lambda_OPL',
                doc='Gain applied to the OPL signal.')
        self._w_OPL = self.shared_parameter(
            lambda x: x.get_config('opl-relative-weight',1.0,float),name='w_OPL')
        I_OPL = self._lambda_OPL * (self._L - self._w_OPL * self._L)

        as_out_state(T.set_subtensor(self._input_init[:,-(input_padded_in_time[:,-(padding[0]):,:,:,:].shape[1]):,:,:,:],
                                    input_padded_in_time[:,-(padding[0]):,:,:,:]), self._input_init)
        super(OPLAllRecursive,self).__init__(make_nd(I_OPL,3),name=name)

class OPLLayerLeakyHeatNode(N):        
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

    Since we want to have some temporal and some spatial convolutions (some 1d, some 2d, but orthogonal to each other), we have to use 3d convolution (we don't have to, but this way we never have to worry about which is which axis). 3d convolution uses 5-tensors (see: <a href="http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d">theano.tensor.nnet.conv</a>), so we define all inputs, kernels and outputs to be 5-tensors with the unused dimensions (color channels and batch/kernel number) set to be length 1.
    """
    def __init__(self,config={},name=None,model=None):
        self.set_config(config)
        
        # center
        self.retina = model
        self.model = model
        
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self.input_variable = make_nd(self.create_input(),5)
        self._E_n_C = self.shared_parameter(
            lambda x: exponential_filter_5d(tau=float(x.get_config('center-tau__sec',0.01)),
                                            n=int(x.get_config('center-n__uint', 0)),
                                            resolution=x.resolution),name='E_n_C')
        self._TwuTu_C = self.shared_parameter(
            lambda x: exponential_highpass_filter_5d(tau=float(x.get_config('undershoot',{}).get('tau__sec',0.1)),
                                                     relative_weight=float(x.get_config('undershoot',{}).get('relative-weight', 0.1)),
                                                     resolution=x.resolution),name='TwuTu_C')
        self._G_C = self.shared_parameter(
            lambda x: gauss_filter_5d(float(x.get_config('center-sigma__deg',0.05)),float(x.get_config('center-sigma__deg',0.05)),
                                      resolution=x.resolution,even=False),name='G_C')
        self._G_S = self.shared_parameter(
            lambda x: gauss_filter_2d(float(x.get_config('surround-sigma__deg',0.15)),float(x.get_config('surround-sigma__deg',0.15)),
                                      resolution=x.resolution,even=False),name='G_S')
        #self._lambda_OPL = self.shared_parameter(
        #    lambda x: x.get_config('opl-amplification',10.0,float) / float(x.model.config.get('input-luminosity-range',255.0)),name='lambda_OPL')
        self._lambda_OPL = self.shared_parameter(
                lambda x: float(x.value_from_config()) / float(self.model.config.get('retina.input-luminosity-range',255.0)),
                save = lambda x: x.value_to_config(float(self.model.config.get('retina.input-luminosity-range',255.0)) * (float(x.var.get_value()))),
                get = lambda x: float(self.model.config.get('retina.input-luminosity-range',255.0)) * (float(x.var.get_value())),
                config_key = 'opl-amplification',
                config_default = 10.0,
                name='lambda_OPL',
                doc='Gain applied to the OPL signal.')
        self._w_OPL = self.shared_parameter(
                lambda x: x.get_config('opl-relative-weight',1.0,float),
                name='w_OPL',
                doc="Weight applied to the surround signal.")

        self._input_init = as_state(dtensor5('input_init'),
                                    init=lambda x: np.zeros((1, get_convis_attribute(self._E_n_C,'update')(x).get_value().shape[1]-1
                                                             + get_convis_attribute(self._TwuTu_C,'update')(x).get_value().shape[1]-1,
                                    1, x.input.shape[1], x.input.shape[2])))
        input_padded_in_time = T.concatenate([
                        self._input_init,
                        self.input_variable],axis=1)
        Nx = self._G_C.shape[3]-1
        Ny = self._G_C.shape[4]-1
        self._L = pad5(pad5(input_padded_in_time,Nx,3),Ny,4)
        self._C = GraphWrapper(make_nd(conv3d(conv3d(conv3d(self._L,self._E_n_C),self._TwuTu_C),self._G_C),3),name='center').graph
        
        # surround
        tau = as_parameter(theano.shared(float(config.get('surround-tau__sec',0.001))),
                           name = 'tau__sec',
                           doc="""$\\tau$ gives the time constant of the exponential decay in seconds.
                           Small values give fast responses while large values give slow responses.
                           The steps to seconds conversion of the associated model will be used to compute.

                           The default value is 10ms.
                           """,
                           initialized = True,
                           optimizable = True,
                           config_key = 'surround-tau__sec',
                           init=lambda x: (x.node.config.get('surround-tau__sec',0.001)))
        steps = as_parameter(theano.shared(model.steps_to_seconds(1.0)),
                            name = 'step',
                            doc="""To convert the time constant in seconds into the appropriate length in bins or steps, this value will be automatically filled via the associatated model.""",
                            initialized = True, 
                            init=lambda x: x.resolution.steps_to_seconds(1.0))
        _preceding_V = as_state(T.dmatrix("preceding_V"),
                               doc="Since recursive filtering needs the result of the previous timestep, the last time step has to be remembered as a state inbetween computations.",
                               init=lambda x: float(x.node.config.get('initial_value',0.5))*np.ones_like(x.input[0,:,:])) # initial condition for sequence
        _preceding_input = as_state(T.dmatrix("preceding_input"),
                               init=lambda x: float(x.node.config.get('initial_value',0.5))*np.ones_like(x.input[0,:,:])) # initial condition for sequence
        a_0 = 1.0
        a_1 = -T.exp(-steps/tau)
        self.a_1 = a_1
        b_0 = 1.0 - a_1
        _k = as_parameter(T.iscalar("k"),init=lambda x: x.input.shape[0]) # number of iteration steps

        ## radial blur
        dtensor4_broadcastable = T.TensorType('float64', (False,False,False,True))
        dtensor3_broadcastable = T.TensorType('float64', (False,False,True))

        kernel = self._G_S
        
        def filter_step(input_image,
                        preceding_V,preceding_input):
            """
                This function computes a single frame for the recursive exponential filtering.

                Additionally, in each step the output is smoothed with a kernel, such that
                activity propagates across the entire population (if given enough time).
            """
            #V = input_image - 0.1*(preceding_input * b_0 - preceding_V * a_1) / a_0
            #V = preceding_V + input_image# + 0.01*(preceding_input * b_0 - preceding_V * a_1) / a_0
            V = (input_image * b_0 - preceding_V * a_1) / a_0

            s0 = (kernel.shape[0]-1)//2
            s0 = (kernel.shape[0]+2)
            s1 = (kernel.shape[1]-1)//2
            s1 = (kernel.shape[1]+2)
            #V_padded = make_nd(pad5(pad5(make_nd(V,5),s0,3,mode = 'const',c=T.mean(V)),s1,4,mode = 'const',c=T.mean(V)),2)
            #V_padded = make_nd(pad5(pad5(make_nd(V,5),s0,3,mode = 'mirror'),s1,4,mode = 'mirror'),2)
            V_padded = pad2_xy(V,s0,s1,mode = 'mirror')
            s0begin = (kernel.shape[0]-1)//2 + s0 -1
            s1begin = (kernel.shape[1]-1)//2 + s1 -1
            s0end = V.shape[0] + s0begin
            s1end = V.shape[1] + s1begin
            V_smoothed = conv2d(V_padded,kernel, border_mode='full')[s0begin:s0end,s1begin:s1end]
            return V_smoothed,input_image
        
        output_variable, _updates = theano.scan(fn=filter_step,
                                      outputs_info=[_preceding_V,_preceding_input],
                                      sequences = [self._C],
                                      non_sequences=[],
                                      n_steps=_k)
        set_convis_attribute(output_variable[0],'name','output')
        as_out_state(output_variable[0][-1],_preceding_V)
        as_out_state(self._C[-1],_preceding_input)
        surround_out = GraphWrapper(output_variable[0],name='surround',ignore=[self._C]).graph
        self._S = surround_out
        I_OPL = self._lambda_OPL * 0.5 * (self._C - self._w_OPL * surround_out)
        
        length_of_filters = self._E_n_C.shape[1]-1+self._TwuTu_C.shape[1]-1
        as_out_state(T.set_subtensor(self._input_init[:,-(input_padded_in_time[:,-(length_of_filters):,:,:,:].shape[1]):,:,:,:],
                                    input_padded_in_time[:,-(length_of_filters):,:,:,:]), self._input_init)
        
        super(OPLLayerLeakyHeatNode,self).__init__(I_OPL,name=name)
        self.node_type = 'OPL Layer LeakyHeat Node'
        self.node_description = lambda: 'Temporal Recursive Filtering and Spatial Convolution'

 
class BipolarLayerNode(N):
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
    def __init__(self,model=None,config=None,name=None,input_variable=None):
        
        self.retina = model
        self.model = model
        self.set_config(config)
        self.state = None
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        # Rewrite with the following assumptions:
        ## controlCond_a is always two elements: a_0, a_1
        ## controlCond_b is always one element: b_0
        # we only need to remember one slice of last input to the inhibitory controlCond
        # we only need to remember one slice of values from the previous timestep

        self._lambda_amp = self.shared_parameter(lambda x: float(x.node.config.get('adaptation-feedback-amplification__Hz',50)),
                                                 name="lambda_amp")
        self._g_leak = self.shared_parameter(lambda x: float(x.node.config.get('bipolar-inert-leaks__Hz',50)),
                                    name="g_leak")
        self._input_amp = self.shared_parameter(lambda x: float(x.node.config.get('opl-amplification__Hz',100)),
                                       name="input_amp")
        self._inputNernst_inhibition = self.shared_parameter(lambda x: float(x.node.config.get('inhibition_nernst',0.0)),
                                                    name="inputNernst_inhibition")

        tau = self.shared_parameter(lambda x: x.resolution.seconds_to_steps(float(x.get_config('adaptation-tau__sec',0.00001))),
                           name = 'tau')
        steps = self.shared_parameter(lambda x: x.resolution.steps_to_seconds(1.0),name = 'step')
        a_0 = 1.0
        a_1 = -T.exp(-steps/tau)
        b_0 = 1.0 - a_1
        # definition of sequences / initial condition for sequences
        self._I_OPL = self.create_input() #sequence
        self._preceding_V_bip = as_state(T.dmatrix("preceding_V_bip"),
            init=lambda x: float(x.node.config.get('initial_value',0.5))*np.ones_like(x.input[0,:,:])) # initial condition for sequence
        self._preceding_inhibition = as_state(T.dmatrix("preceding_inhibition"),
            init=lambda x: float(x.node.config.get('initial_value',0.5))*np.ones_like(x.input[0,:,:])) # initial condition for sequence
        self._inhibition_smoothing_kernel = self.shared_parameter(
            lambda x: gauss_filter_2d(float(x.get_config('adaptation-sigma__deg',0.2)),float(x.get_config('adaptation-sigma__deg',0.2)),
                                      resolution=x.resolution,even=False),
            name='inhibition_smoothing_kernel')
                #T.dmatrix(self.name+"_inhibition_smoothing_kernel") # initial condition for sequence
        self._k_bip = as_parameter(T.iscalar("k"),init=lambda x: x.input.shape[0]) # number of iteration steps

        # Definition of bipolar computations in an iterative loop
        #    compare with the implementation in python for a version
        #    that is closer to virtual retina.
        #
        # The simulation has two populations of neurons, each being the size of the input image.
        # `V_bip` is the output of this stage and the excitatory population (called `daValues` in VR)
        #   `V_bip` recieves current input through `input_image` and conductance input through `preceding_inhibition`
        # `inhibition` is the inhibitory population (`controlCond` in VR)
        #   it gets a E and G filtered version of the rectified V_bip
        #
        # The `attenuation_map` provides a different conductance for each pixel 
        #    it is calculated from the previous time steps `inhibition` value
        #    which in turn is a filtered version of the rectified `V_bip`.
        # The shape of the rectification is controlled by `g_leak` and `lambda_amp`
        def bipolar_step(input_image,
                        preceding_V_bip, preceding_attenuationMap, preceding_inhibition, 
                        lambda_amp, g_leak, input_amp,inputNernst_inhibition,inhibition_smoothing_kernel):
                        # note: preceding_attenuationMap is not used: this is only an output
            total_conductance = as_variable(g_leak + as_variable(preceding_inhibition,name='preceding_inhibition'),'total_conductance')
            attenuation_map = as_variable(T.exp(-steps*total_conductance),'attenuation map')
            E_infinity = as_variable((input_amp * as_variable(input_image,name='input_image') + inputNernst_inhibition * preceding_inhibition)/total_conductance,name='E_infinity')
            V_bip = as_variable(((preceding_V_bip - E_infinity) * attenuation_map) + E_infinity,name='V_bip') # V_bip converges to E_infinity
            
            s0 = (inhibition_smoothing_kernel.shape[0]-1)/2
            s0end = preceding_V_bip.shape[0] + s0
            s1 = (inhibition_smoothing_kernel.shape[1]-1)/2
            s1end = preceding_V_bip.shape[1] + s1
            inhibition = as_variable((conv2d((lambda_amp*(preceding_V_bip)**2 * b_0 
                                       - preceding_inhibition * a_1) / a_0, inhibition_smoothing_kernel, border_mode='full')[s0:s0end,s1:s1end]),'smoothed_inhibition')
            # // # missing feature from Virtual Retina:
            # // ##if(gCoupling!=0)
            # // ##  leakyHeatFilter.radiallyVariantBlur( *targets ); //last_values...

            return [V_bip,attenuation_map,inhibition]

        # The order in theano.scan has to match the order of arguments in the function bipolar_step
        self._result, self._updates = theano.scan(fn=bipolar_step,
                                      outputs_info=[(self._preceding_V_bip),T.zeros_like(self._preceding_V_bip),(self._preceding_inhibition)],
                                      sequences = [self._I_OPL],
                                      non_sequences=[self._lambda_amp, self._g_leak, self._input_amp,
                                                     self._inputNernst_inhibition, self._inhibition_smoothing_kernel],
                                      n_steps=self._I_OPL.shape[0])#self._k_bip)
        as_out_state(self._result[0][-1],self._preceding_V_bip)
        # attenuation is not part of the state, but can be used as an output sequence
        as_out_state(self._result[2][-1],self._preceding_inhibition)
        ## The order of arguments presented here is arbitrary (will be inferred by the symbols provided),
        ##  but function calls to compute_V_bip have to match this order!
        #self.compute_V_bip = theano.function(inputs=[self._I_OPL,self._preceding_V_bip,self._preceding_inhibition,
        #                                              self._lambda_amp, self._g_leak, self._step_size, self._input_amp,
        #                                              self._inputNernst_inhibition, self._inhibition_smoothing_kernel, self._b_0, self._a_0, self._a_1,
        #                                              self._k_bip], 
        #                                      outputs=self._result, 
        #                                      updates=self._updates)
        super(BipolarLayerNode,self).__init__(make_nd(self._result[0],3),name=name)


class GanglionInputLayerNode(N):
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
    def __init__(self,model=None,config=None,name=None,input_variable=None):
        
        self.retina = model
        self.model = model
        self.set_config(config)
        self.state = None
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self._V_bip = make_nd(self.create_input(),5)
        # TODO: state? what about previous episode? concatenate?
        #num_V_bip = input.reshape((1,input.shape[0],1,input.shape[1],input.shape[2]))
        self._T_G = self.shared_parameter(lambda x: float(x.get_config('sign',1)) * 
                                                        exponential_highpass_filter_5d(tau=float(x.get_config('transient-tau__sec',0.04)),
                                                            relative_weight=float(x.get_config('transient-relative-weight',0.75)),
                                                            resolution=x.resolution),
                                     name = 'T_G')
        self._input_init = as_state(dtensor5('input_init'),
                                    init=lambda x: np.zeros((1, get_convis_attribute(x.node._T_G,'update')(x).get_value().shape[1]-1,1, x.input.shape[1], x.input.shape[2])))

        #self._V_bip_padded = T.concatenate([T.zeros((1,self._T_G.shape[1]-1,1,self._V_bip.shape[3],self._V_bip.shape[4])),self._V_bip],axis=1)
        self._V_bip_padded = as_variable(T.concatenate([self._input_init,self._V_bip],axis=1),'V_bip_padded')

        length_of_filters = self._T_G.shape[1]-1
        as_out_state(T.set_subtensor(self._input_init[:,-(self._V_bip_padded[:,-(length_of_filters):,:,:,:].shape[1]):,:,:,:],
                                    self._V_bip_padded[:,-(length_of_filters):,:,:,:]), self._input_init)

        self._V_bip_E = as_variable(conv3d(self._V_bip_padded,self._T_G),'V_bip_E')
        self._i_0_G = self.shared_parameter(lambda x: float(x.get_config('value-at-linear-threshold__Hz',70.0)),
                                          name="i_0_G")
        self._v_0_G = self.shared_parameter(lambda x: float(x.get_config('bipolar-linear-threshold',0.0)),
                                          name="v_0_G")
        self._lambda_G = self.shared_parameter(lambda x: float(x.get_config('bipolar-amplification__Hz',100.0)),
                                          name="lambda_G")
        self._G_gang = self.shared_parameter(lambda x: gauss_filter_5d(float(x.get_config('sigma-pool__deg',0.0)),float(x.get_config('sigma-pool__deg',0.0)),
                                                                       resolution=x.resolution,even=False),
                                        name = 'G_gang')
        self._N = GraphWrapper(as_variable(theano.tensor.switch(self._V_bip_E < self._v_0_G, 
                                 as_variable(self._i_0_G/(1-self._lambda_G*(self._V_bip_E-self._v_0_G)/self._i_0_G),name='N_0',html_name='N<sub>V&lt;v0</sub>'),
                                 as_variable(self._i_0_G + self._lambda_G*(self._V_bip_E-self._v_0_G),name='N_1',html_name='N<sub>V&gt;=v0</sub>')),'N_G_gang',
                        requires=[self._lambda_G,self._i_0_G,self._v_0_G]),name='N',ignore=[self._V_bip_E]).graph

        #self.compute_N = theano.function([self._V_bip, self._T_G, self._i_0_G, self._v_0_G, self._lambda_G], self._N)

        self._I_Gang = conv3d(self._N,self._G_gang)
        self.output_variable = self._I_Gang
        #self.compute_I_Gang = theano.function([self._V_bip, self._T_G, self._i_0_G, self._v_0_G, self._lambda_G, self._G_gang], theano.Out(self.output_variable, borrow=True))
        super(GanglionInputLayerNode,self).__init__(make_nd(self._I_Gang,3),name=name)
    def __repr__(self):
        return '[Ganglion Input Node] Shape: '+str(fake_filter_shape(self._G_gang.get_value(),self._T_G.get_value()))

class GanglionSpikingLayerNode(N):
    """
    **TODO:DONE** ~~The refractory time now working!~~

    The ganglion cells recieve the gain controlled input and produce spikes. 

    When the cell is not refractory, :math:`V` moves as:

    $$ \\\\dfrac{ dV_n }{dt} = I_{Gang}(x_n,y_n,t) - g^L V_n(t) + \eta_v(t)$$

    Otherwise it is set to 0.
    """
    def __init__(self,model=None,config=None,name=None,input_variable=None):
        self.retina = model
        self.model = model
        self.set_config(config)
        self.state = None
        self.last_noise_slice = None
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self._k_gang = as_parameter(T.iscalar("k_gang_spike"),init=lambda x: x.input.shape[0], doc='Length of the input sequence (set automatically).')
        
        #obsolete? self._refrac = T.dscalar(name+"refrac")
        
        self.input_variable = self.create_input()
        self.input_padding = as_state(T.dtensor3("initial_I"), init=lambda x: x.input[:1,:,:])
        self._I_gang = as_variable(T.concatenate([self.input_padding, self.input_variable]),'I_gang') # input
        
        self._initial_refr = as_state(
                T.dmatrix("initial_refr"),
                init=lambda x: np.zeros_like(x.input[0,:,:]) 
                        if x.node.config.get('random-init',True) is False
                        else (x.resolution.seconds_to_steps(x.node.config.get('refr-mean__sec',0.0005))*np.random.rand(*x.input[0,:,:].shape)),
                doc="Initialization of the refractory times. If `random-init` is `True`, each cell gets a random value between `[0..refr-mean__sec]`"
                )
        self._V_initial = as_state(
                T.dmatrix("V_initial"),
                doc='The initial state of the membrane potential (can be randomized with `random-init`).',
                init=lambda x: np.zeros_like(x.input[0,:,:])
                                if x.node.config.get('random-init',True) is False
                                else np.random.rand(*x.input[0,:,:].shape))

        self._refr_sigma = self.shared_parameter(
                lambda x: float(x.resolution.seconds_to_steps(float(x.value_from_config()))),
                save = lambda x: x.value_to_config(x.resolution.steps_to_seconds(float(x.var.get_value()))),
                get = lambda x: float(x.resolution.steps_to_seconds(float(x.var.get_value()))),
                config_key = 'refr-stdev__sec',
                config_default = 0.001,
                name='refr_sigma',
                doc='The standard deviation of the refractory time that is randomly drawn around `refr_mu`')

        self._refr_mu = self.shared_parameter(
                lambda x: float(x.resolution.seconds_to_steps(float(x.value_from_config()))),
                save = lambda x: x.value_to_config(x.resolution.steps_to_seconds(float(x.var.get_value()))),
                get = lambda x: float(x.resolution.steps_to_seconds(float(x.var.get_value()))),
                config_key = 'refr-mean__sec',
                config_default = 0.000523,
                name='refr_mu',
                doc="The mean of the distribution of random refractory times (in seconds).")

        self._g_L = self.shared_parameter(
                lambda x: float(x.value_from_config()),
                config_key = 'g-leak__Hz',
                config_default = 10,
                doc='Leak current (in Hz or dimensionless firing rate).',
                name = 'g_L')

        self._raw_noise_gang = as_parameter(
                T.dtensor3("noise_gang"),
                init=lambda x: np.random.randn(*x.input.shape),
                doc ='Random input (will be scaled by sigma-V and refr_stdev). Since the membrane potential already crossed the threshold when we want to draw a refractory period, we can safely use the next sample from this random input without introducing any codependency.')

        self._noise_state = as_state(T.dtensor3("initial_noise"), init=lambda x: np.random.randn(*x.input[:1,:,:].shape))
        self._noise_gang = T.concatenate([self._noise_state,self._raw_noise_gang]) # we need to remember one noise state
        self._noise_sigma = self.shared_parameter(
                lambda x: float(x.value_from_config())*np.sqrt(2*x.resolution.seconds_to_steps(float(x.get_config('g-leak__Hz',10)))),
                save = lambda x: x.value_to_config(float(x.var.get_value())/np.sqrt(2*x.model.seconds_to_steps(float(x.get_config('g-leak__Hz',10))))),
                get = lambda x: (x.var.get_value())/np.sqrt(2*x.resolution.seconds_to_steps(float(x.get_config('g-leak__Hz',10)))),
                config_key = 'sigma-V',
                config_default = 0.1,
                doc='Amount of noise added to the membrane potential.',
                name = "noise_sigma")

        #self.random_init = self.config.get('random-init',None)
        
        self._tau = shared_parameter(lambda x: x.resolution.steps_to_seconds(1.0),
                O()(node=self,model=self.model),
                doc = 'Length of timesteps (ie. the steps_to_seconds(1.0) of the model.',
                name = 'tau')
        def spikeStep(I_gang, noise_gang,noise_gang_prev,
                      prior_V, prior_refr,  
                      noise_sigma, refr_mu, refr_sigma, g_L,tau_gang):
            V = prior_V + (I_gang - g_L * prior_V + noise_sigma*(noise_gang))*tau_gang
            V = as_variable(theano.tensor.switch(T.gt(prior_refr, 0.5), 0.0, V),'V')
            spikes = T.gt(V, 1.0)
            refr = as_variable(theano.tensor.switch(spikes,
                    prior_refr + refr_mu + refr_sigma * noise_gang,
                    prior_refr - 1.0
                    ),'refr')
            next_refr = theano.tensor.switch(T.lt(refr, 0.0),0.0,refr)
            return [V,next_refr,spikes]
        k = self.input_variable.shape[0]
        self._result, updates = theano.scan(fn=spikeStep,
                                      outputs_info=[(self._V_initial),(self._initial_refr),None],
                                      sequences = [self._I_gang,dict(input=self._noise_gang, taps=[-0,-1])],
                                      non_sequences=[self._noise_sigma, self._refr_mu, self._refr_sigma, self._g_L, self._tau],
                                      n_steps=k)#self._k_gang)
        as_out_state(self._I_gang[-1:,:,:],self.input_padding) #self.input_variable
        as_out_state(self._noise_gang[-1:,:,:],self._noise_state)
        as_out_state(self._result[0][-1],self._V_initial)
        as_out_state(self._result[1][-1],self._initial_refr)
        self.output_V = as_variable(self._result[0],name='output_V',
            doc='The membrane potential of each unit for each time step.')
        self.output_refractory = as_variable(self._result[1],name='output_refractory',
            doc='length of refractory period for each unit until it is allowed to spike again.')
        self.output_spikes = as_variable(self._result[2],name='output_spikes',
            doc='Binary count of spikes for each unit for each timestep. The maximal firing rate of this spike generation process is bounded by the size of time bins.')
        #spikes = T.extra_ops.diff(T.gt(self._result[1],0.0),n=1,axis=0)
        super(GanglionSpikingLayerNode,self).__init__(self._result[2],name=name)
    def __repr__(self):
        return '[Ganglion Spike Node] Differential Equation'
