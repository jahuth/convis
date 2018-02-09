"""

Optimizer classes in addition to the ones provided by
`torch.optim`.

The Optimizers used here assume that they estimate one 
set of parameters. If the model should be fitted to some
data at one time and to other data at another time, a new
instance of the optimizer should be used.

You can set the optimizer of a model directly for that:

.. code::
    python

    import convis
    m = convis.LNLN()
    m.set_optimizer.LBFGS()
    m.optimize(input_a, goal_a)
    a_optim = m._optimizer # store the optimizer 
    m.set_optimizer.LBFGS() # initialize a new optimizer
    m.optimize(input_b, goal_b) # optimizing with the new optimizer
    m._optimizer = a_optim # using the first optimizer again

But this method can leave the optimizer confused (ie. it might not
work as intended), as state of the model and the parameters are
changed by running the second optimizer on some other input.

To use the same model for two different fitting processes
for two different processes that have to be estimated,
it is recommended to backup all relevant information and
to restore it when returning to fitting a previous process.

To do that there are three options:
    - using `v = model.get_all()` to retrieve the information into a variable and `model.set_all(v)` to restore it
    - using `model.push_all()` to push the information onto a stack within the model and `model.pop_all()` to retrieve it. With this method the values  can only be restored once, unless pushed again onto the stack.
    - using `model.store_all(some_name)` to store the information under a certain name and retrieving it with `model.retrieve_all(some_name)`, which can be used more than once and does not rely on user managed variables.

.. code::
    python

    import convis
    m = convis.LNLN()
    m.store_all('init') # stores state, parameter values and optimizer under a name
    m.set_optimizer.LBFGS()
    m.optimize(input_a, goal_a)
    m.push_all() # alternatively, you can save the optimizer, 
    # state and parameters onto a stack (optimizers will 
    # mostly assume that the parameters are not changed
    # between steps, but this differs per algorithm)
    m.retrieve_all('init') # retrieves state, parameter values and optimizer from before
    m.set_optimizer.LBFGS() # initialize a new optimizer
    m.optimize(input_b, goal_b) # optimizing with the new optimizer
    m.pop_all() # returning to the previous parameters, state and optimizer

"""

from torch.optim.optimizer import Optimizer
from collections import defaultdict
import numpy as np
import torch

class FiniteDifferenceGradientOptimizer(Optimizer):
    """
        Quasi-Newton method with a finite difference approximation
        of 2nd order gradient.
    """
    def __init__(self, params, **kwargs):
        defaults = kwargs
        self.grads = defaultdict(list)
        self.values = defaultdict(list)
        super(FiniteDifferenceGradientOptimizer, self).__init__(params, defaults)        
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                value = p.data
                grad = p.grad.data
                self.grads[p].append(grad.numpy().copy())
                self.values[p].append(value.numpy().copy())
                
                if len(self.grads[p]) > 1:
                    estimate = np.zeros_like(self.values[p][-1])
                    normalization_term = 0.0
                    for i in range(1,len(self.values[p])):
                        x0 = (self.values[p][i-1] + self.grads[p][i-1]
                                                  * (self.values[p][i-1]-self.values[p][i])
                                                  / (self.grads[p][i]-self.grads[p][i-1])
                             )
                        weight = np.sqrt(np.nanmean((self.grads[p][i]-self.grads[p][i-1])**2))
                        if np.isnan(weight) or weight in [np.inf, np.nan] or weight < 0.0001:
                            # if weight is wrong, we won't deal with this
                            continue
                        #print weight
                        x0[self.grads[p][-1] == self.grads[p][-2]] = 0.0
                        x0[np.isnan(x0)] = 0.0
                        estimate += x0.reshape(value.shape)*weight
                        normalization_term += weight
                    p.data = torch.Tensor(estimate/float(normalization_term))
                    #else:
                    #    print "Last values are the same!"
                else:
                    estimate = value - 0.1 * grad
                    p.data.add_(-grad/grad.std())
        return loss

class CautiousLBFGS(Optimizer):
    """
        Executes the LBFGS optimizer, but chooses new starting
        values if the method is instable due to the closeness
        to the true value.
    """
    def __init__(self, params, **kwargs):
        defaults = kwargs
        self.grads = defaultdict(list)
        self.values = defaultdict(list)
        super(FiniteDifferenceGradientOptimizer, self).__init__(params, defaults)        
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                value = p.data
                grad = p.grad.data
                self.grads[p].append(grad.numpy().copy())
                self.values[p].append(value.numpy().copy())