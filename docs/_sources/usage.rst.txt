Usage
=====

Running a model
-----------------

.. code-block:: python

    import convis
    retina = convis.retina.Retina()
    retina(some_short_input)
    retina.run(some_input,dt=100)

Usually PyTorch Layers are callable and will perform their forward computation when called with some input. But since Convis deals with long (potentially infinite) video sequences, a longer input can be processed in smaller chunks by calling `Layer.run(input,dt=..)` with `dt` set to the length of input that should be processed at a time. This length depends on the memory available in your system and also if you are using the model on your cpu or gpu.
`.run` also accepts numpy arrays as input, which will be converted into PyTorch `Tensor`s and packaged as a `Variable`.


Switching between CPU and GPU usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch objects can move between GPU memory and RAM by calling `.cuda()` and `.cpu()` methods respectively. This can be done on a single Tensor or on an entire model.


Using Runner objects
~~~~~~~~~~~~~~~~~~~~

Runner objects can execute a model on a fixed set of input and output streams. 
The execution can also happen in a separate thread:

.. code-block:: python

    import convis, time
    import numpy as np

    inp = convis.streams.RandomStream(size=(10,10),pixel_per_degree=1.0,level=100.2,mean=128.0)
    out1 = convis.streams.SequenceStream(sequence=np.ones((0,10,10)), max_frames=10000)
    retina = convis.retina.Retina()
    runner = convis.base.Runner(retina, input = inp, output = out1)
    runner.start()
    time.sleep(5) # let thread run for 5 seconds or longer
    plot(out1.sequence.mean((1,2)))
    # some time later
    runner.stop()


Optimizing a Model
--------------------

One way to optimize a model is by using the `.set_optimizer` attribute and the `.optimize` method:

.. code-block:: python

    l = convis.models.LN()
    l.set_optimizer.SGD(lr=0.001) # selects an optimizer with arguments
    #l.optimize(some_inp, desired_outp) # does the optimization with the selected optimizer


A full example:

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import convis
    import torch
    l_goal = convis.models.LN()
    k_goal = np.random.randn(5,5,5)
    l_goal.conv.set_weight(k_goal)
    plt.plot(l_goal.conv.weight.data.cpu().numpy()[0,0,:,:,:].mean(1))
    plt.matshow(l_goal.conv.weight.data.cpu().numpy().mean((0,1,2)))
    plt.colorbar()
    l = convis.models.LN()
    #l.conv.set_weight(np.ones((5,5,5)),normalize=True)
    #l.set_optimizer.LBFGS()
    #l.cuda()
    #l_goal.cuda()
    #inp = 1.0*(np.random.randn(200,10,10))
    #inp = torch.autograd.Variable(torch.Tensor(inp)).cuda()
    #outp = l_goal(inp[None,None,:,:,:])
    #plt.figure()
    #plt.plot(l_goal.conv.weight.data.cpu().numpy()[0,0,:,:,:].mean(1),'--',color='red')
    #for i in range(50):
    #    l.optimize(inp[None,None,:,:,:],outp)
    #    if i%10 == 2:
    #        plt.plot(l.conv.weight.data.cpu().numpy()[0,0,:,:,:].mean(1))
    #plt.matshow(l.conv.weight.data.cpu().numpy().mean((0,1,2)))
    #plt.colorbar()
    #plt.figure()
    #h = plt.hist((l.conv.weight-l_goal.conv.weight).data.cpu().numpy().flatten(),bins=15)




.. plot::

    import numpy as np
    import matplotlib.pylab as plt
    import convis
    import torch
    l_goal = convis.models.LN()
    k_goal = np.random.randn(5,5,5)
    l_goal.conv.set_weight(k_goal)
    plt.plot(l_goal.conv.weight.data.cpu().numpy()[0,0,:,:,:].mean(1))
    plt.matshow(l_goal.conv.weight.data.cpu().numpy().mean((0,1,2)))
    plt.colorbar()
    l = convis.models.LN()
    l.conv.set_weight(np.ones((5,5,5)),normalize=True)
    l.set_optimizer.LBFGS()
    l.cuda()
    l_goal.cuda()
    inp = 1.0*(np.random.randn(200,10,10))
    inp = torch.autograd.Variable(torch.Tensor(inp)).cuda()
    outp = l_goal(inp[None,None,:,:,:])
    plt.figure()
    plt.plot(l_goal.conv.weight.data.cpu().numpy()[0,0,:,:,:].mean(1),'--',color='red')
    for i in range(50):
        l.optimize(inp[None,None,:,:,:],outp)
        if i%10 == 2:
            plt.plot(l.conv.weight.data.cpu().numpy()[0,0,:,:,:].mean(1))
    plt.matshow(l.conv.weight.data.cpu().numpy().mean((0,1,2)))
    plt.colorbar()
    plt.figure()
    h = plt.hist((l.conv.weight-l_goal.conv.weight).data.cpu().numpy().flatten(),bins=15)


When selecting an Optimizer, the full list of available Optimizers can be seen by tab-completion.

Some interesting optimizers are:

  * SGD: Stochastic Gradient Descent - one of the simplest possible methods, can also take a momentum term as an option
  * Adagrad/Adadelta/Adam/etc.: Accelerated Gradient Descent methods - adapt the learning rate
  * LBFGS: Broyden-Fletcher–Goldfarb-Shanno (Quasi-Newton) method - very fast for many almost linear parameters

Using an Optimizer by Hand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The normal PyTorch way to call Optimizers is to fill the gradient buffers by hand and then calling `.step()` (see also http://pytorch.org/docs/master/optim.html ).

.. code-block:: python

    import numpy as np
    import convis
    import torch
    l_goal = convis.models.LN()
    k_goal = np.random.randn(5,5,5)
    l_goal.conv.set_weight(k_goal)
    inp = 1.0*(np.random.randn(200,10,10))
    inp = torch.autograd.Variable(torch.Tensor(inp)).cuda()
    outp = l_goal(inp[None,None,:,:,:])
    l = convis.models.LN()
    l.conv.set_weight(np.ones((5,5,5)),normalize=True)
    optimizer = torch.optim.SGD(l.parameters(), lr=0.01)
    for i in range(50):
        # first the gradient buffer have to be set to 0
        #optimizer.zero_grad()
        # then the computation is done
        o = l(inp)
        # and some loss measure is used to compare the output to the goal
        loss = ((outp-o)**2).mean() # eg. mean square error
        # applying the backward computation fills all gradient buffers with the corresponding gradients
        #loss.backward(retain_graph=True)
        # now that the gradients have the correct values, the optimizer can perform one optimization step
        #optimizer.step()

Or using a closure function, which is necessary for advanced optimizers that need to re-evaluate the loss at different parameter values:

.. code-block:: python

    l = convis.models.LN()
    l.conv.set_weight(np.ones((5,5,5)),normalize=True)
    optimizer = torch.optim.LBFGS(lr=0.01)

    def closure():
        optimizer.zero_grad()
        o = l(inp)
        loss = ((outp-o)**2).mean()
        loss.backward(retain_graph=True)
        return loss

    #for i in range(50):
    #    optimizer.step(closure)


The `.optimize` method of `convis.Layer`s does exactly the same as the code above. It is also possible to supply it with alternate optimizers and loss functions:

.. code-block:: python

    l = convis.models.LN()
    l.conv.set_weight(np.ones((5,5,5)),normalize=True)
    opt2 = torch.optim.LBFGS(l.parameters())
    #l.optimize(inp[None,None,:,:,:],outp, optimizer=opt2, loss_fn = lambda x,y: (x-y).abs().sum()) # using LBFGS (without calling .set_optimizer) and another loss function

`.set_optimizer.*()` will automatically include all the parameters in the model, if no generator/list of parameters is used as the first argument. 