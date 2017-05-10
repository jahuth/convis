Usage
=====

The convis toolbox provides a framework for creating models of visual processes.
The toolbox can be used on three different levels:

  1. by running a model and tuning its parameters
  2. Creating a custom model out of `Layers`
  3. Creating a custom Layer to be used in a model

Self-help
-----------------

`convis.describe(...)` will give you information about any object of you don't know what it does.
Also most objects contain doc strings which can be accessed in most python environments.
In the case of theano variables, the doc string is theano specific, but with `convis.help(...)`
the convis specific documentation to this specific variable can be retrieved.



Running a model
-----------------


.. note::

    The size of images and the length of time you can process in one go will depend on your system and specifically on the amount of continouus memory available on your graphics card. If theano can not find enough continouus memory, the python code running the model will fail and until this python process is restarted it might hold some GPU memory as unusable.
    For now there is no good way to fix this other than restarting the process and choosing a smaller chunk size.

The simplest case of running a model is to instanciate the model and call the `.run` method::

    In [1]: import convis, numpy
    In [2]: retina = convis.retina.Retina()
    In [3]: o = retina.run(numpy.zeros((200,10,10)))
    In [4]: convis.describe(o) # optional visualization of the output

When the retina model is executed, the model is compiled to run on your GPU 
(if available) or CPU otherwise and the output(s) of the model will be saved into
`o`, an Output object.

The retina model produces more than one output. To visualize where this output is coming from we can 
print a simplified version of the model::

    In [1]: print retina.draw_simple_diagram()

        input -> OPL -> Bipolar -> GanglionInputLayer_Parvocellular_On -> GanglionSpikes__Parvocellular_On -> output
                                -> GanglionInputLayer_Parvocellular_Off -> GanglionSpikes__Parvocellular_Off -> output

After the Layer "Bipolar", the model is split into two pathways, one for On, the other for Off responses.
Each of these has its own spiking output.

The content of output objects can be addressed in one of two ways: by name or by numerical index::

    In [1]: o.GanglionSpikes__Parvocellular_On_output_spikes is o[0]
    Out[1]: True
    In [2]: o.GanglionSpikes__Parvocellular_Off_output_spikes is o[1]
    Out[2]: True



Changing Parameters
~~~~~~~~~~~~~~~~~~~~~~

When creating a new model from an xml or json configuration, the easiest way is to change this configuration.
But once the model is created, it is easier to directly access the variables.

.. note::

    Updating shared variables in an existing model from a changed configuration is not implemented right now. So changing the values directly is the only option when the model is already created.
    This is different for input parameters (see below for the difference between them).

Since no one should be forced to remember the fairly arbitrary names of the 
different parameters, they act as attributes such that Tab completion (eg. in an ipython console
or a ipython/jupyter notebook) gives you a list of all names.

By default this list is categorized by layers (and sub-layers)::

    In[1]: retina = convis.retina.Retina()
    In[2]: retina.parameters.<press tab>
                retina.parameters.Bipolar 
                retina.parameters.GanglionInputLayer_Parvocellular_Off
                retina.parameters.GanglionInputLayer_Parvocellular_On
                retina.parameters.GanglionSpikes__Parvocellular_Off
                retina.parameters.GanglionSpikes__Parvocellular_On
                retina.parameters.OPL

    In[3]: retina.parameters.OPL.<press tab>
                retina.parameters.OPL.center
                retina.parameters.OPL.surround
                retina.parameters.OPL.lambda_OPL
                retina.parameters.OPL.w_OPL
                retina.parameters.OPL.Reshape_C_S

    In[4]: retina.parameters.OPL.center.<press tab>
                retina.parameters.OPL.center.E_n_C
                retina.parameters.OPL.center.G_C
                retina.parameters.OPL.center.Twu_Tu_C

To see all parameters in a flat list instead of the nested structure, use the special attribute `_all`::

    In[5]: retina.parameters._all.<press tab>
		# a very long list of all parameters

Since these names still might not give you an idea of what they do, you can let convis describe them::

    In [6]: convis.describe(retina.parameters.OPL.center.E_n_C)


    Out [6]: {'doc': 'The n-fold cascaded exponential creates a low-pass characteristic.
                      A filter can be created with `retina_base.m_en_filter`\n',
     'name': 'E_n_C',
     'node': [Node] center: ,
     'simple_name': 'E_n_C',
     'value': array([[[[[ 0.63214926]]],
             [[[ 0.23255472]]],
             [[[ 0.0855521 ]]],
             [[[ 0.03147286]]],
             [[[ 0.01157822]]],
             [[[ 0.00425939]]],
             [[[ 0.00156694]]]]])}


´describe´ will output a formatted table in an ipython notebook and a dictionary in a console.
To verify that it is actually a shared parameter, we can ask for its type::

    In [7]: type(retina.parameters.OPL.center.E_n_C)

    Out [7]: theano.tensor.sharedvar.TensorSharedVariable

So what we learned is that `retina.parameters.OPL.center.E_n_C` is a convolution kernel and has some specific value. It is a theano share variable, which means that it represents both, a node in a computational graph as well as a portion of memory which is synchronized between RAM and GPU memory. To change the value we can use the method `set_value()`::

    k = convis.numerical_filters.exponential_filter_5d(tau=0.02),resolution=retina.resolution)
    retina.parameters.OPL.center.E_n_C.set_value(k)

For parameters which are accessed through the `M().parameters.` attribute, there is also a shortcut for setting the value::

    k = convis.numerical_filters.exponential_filter_5d(tau=0.02),resolution=retina.resolution)
    retina.parameters.OPL.center.E_n_C = k

.. note:

    This will only work if the parameter is accessd through the `parameter` attribute structure. If the reference of the theano variable is eg. saved to a python variable, using `some_var = new_value` will not change the value of the thenao variable!



Creating a model from layers
----------------------------


Initializing a Layer
~~~~~~~~~~~~~~~~~~~~~~

Layers typically expect a configuration dictionary that contains the initial values of their
parameters.

Initializing a convolution filter::

    k = convis.numerical_filters.gauss_filter_5d()
    # creates a 5 dimensional tensor with a 2d gaussian filter
    # in the x and y dimensions
    gauss_filter_layer = convis.filters.simple.K_5d_kernel_filter(config = {
            'kernel': k
        })


`K_5d_kernel_filter` is a layer that computes a convolution of the input with a 5 dimensional 
kernel filter. The dimensions are "batch", "time", "channel", "x", "y".

TODO: make a section for convolution filters

Connecting Layers
~~~~~~~~~~~~~~~~~~~~~~

Layers can either be connected one by one by using `add_input` or in batch by using `convis.connect`.
A shorthand for `b.addinput(a)` is `b += a`.
The following is equivalent::

    convis.connect([a,b,c])

    c.add_input(b)
    b.add_input(a)

    c += b # if c and b are convis Layers
    b += a
    
Both will result in a graph in which only `a` has an open input. `b` recieves the output of `a` as input, `c` recieves the output of b.

If more than one input or output are specified, a specific one can be set like this::

    b.add_output(a.graph[0],'input_0') 
    # connects the first element of the 
    # output of a to the input named 'input_0' of b
    b.add_output(a.graph[1],'input_1')
    b.add_output(a.graph[2],'input_2')

If two outputs are connected to the same input, their values will be added together.

More complicated connectivity can be achieved in `convis.connect` by nesting lists::

    convis.connect([a,[b,c,[d,e,f]],g])
    
    # is equivalent to:
    b.add_input(a) 
            # the second nested level is even, 
            # so connected in parallel
    c.add_input(a)
    d.add_input(a)
    e.add_input(d) 
            # the third nested level is odd, 
            # so again connected in sequence
    f.add_input(e) 
    g.add_input(b) 
            # after a parallel nesting is closed, 
            # the inputs are all summed 
            # in the next layer
    g.add_input(c)
    g.add_input(f)



Creating a layer from a graph
-----------------------------

Example::

    class Delay(convis.N):
        def __init__(self,config,name=None,model=None):
            inp = self.create_input()
            delay = config.get('delay',1)
            self._input_init = convis.as_state(convis.T.dtensor3('input_init'),
                                        init=lambda x: np.zeros((delay,
                                        x.input.shape[1], x.input.shape[2])))
            o = T.concatenate([self._input_init,
                               inp[delay:,:,:]],axis=0)
            convis.as_out_state(T.set_subtensor(self._input_init[-(o[-(delay):,:,:].shape[0]):,:,:],
                                        o[-(delay):,:,:]), self._input_init)
            super(Delay,self).__init__(o,name=name)



Two kinds of parameters
~~~~~~~~~~~~~~~~~~~~~~~~~


