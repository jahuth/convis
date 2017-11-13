from torch.optim.optimizer import Optimizer
from collections import defaultdict
import numpy as np
import torch

class FiniteDifferenceGradientOptimizer(Optimizer):
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