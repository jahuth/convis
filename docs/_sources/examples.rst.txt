.. _examples:

.. convis documentation master file, created by
   sphinx-quickstart on Wed Dec  7 18:52:26 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Examples
========




Running a Model
----------------

An LN Filter::

    import convis
    the_input = convis.samples.moving_bars(2000)
    
    m = convis.models.LN()   
    o = m.run(the_input)


The premade retina model can be instanciated and executed like this::

    import convis
    m = convis.retina.Retina()
    m.run(the_input)


If the input is very long, it can be broken into chunks::

    m.run(the_input,dt=1000) # takes 1000 timesteps at a time

A runner runs in its own thread and consumes an input stream::

    import convis
    model = convis.retina.Retina()
    input_stream = convis.streams.RandomStream(size=(50,50))
    output_stream = convis.streams.InrImageStreamWriter('output.inr')
    runner = convis.Runner(model,input_stream,output_stream)
    runner.start()

The input stream can be infinite in length (as eg. the `RandomStream`), see :mod:`convis.streams`.

More information specific to the retina model can be found :ref:`here <model_retina>`.


Indices
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

