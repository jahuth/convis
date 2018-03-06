"""
Analysis Tools
--------------



"""
import numpy as np
import torch
import copy
__all__ = ['sta','stc','LinearGradientExplorer']

def sta(stim,spikes,filter_shape=(10,1,1),binary=True,threshold=0.5):
    """Calculates a Spike Triggered Average stimulus
        
        Parameters
        ----------
    
        stim (np.array(time,x,y)):
            stimulus presented
        spikes (np.array(time,x,y)):
            spikes recorded
        filter_shape (tuple)
            The desired output filter shape
        binary (bool)
            Whether the spikes are already a binary sequence (0s and 1s or True and False)
        threshold (float)
            If binary is False, this threshold converts the input array into
            a binary array

        Notes
        -----
        For now we ignore spikes too close to the edge!
    """
    if binary:
        spikes = np.array(np.where(spikes)).transpose()
    else:
        spikes = np.array(np.where(spikes>=threshold)).transpose()
    avg_sum = np.zeros(filter_shape)
    avg_count = len(spikes)
    for s in spikes:
        trigger = stim[s[0]:s[0]+filter_shape[0],
                        s[1]:s[1]+filter_shape[1],
                        s[2]:s[2]+filter_shape[2]]
        if trigger.shape == avg_sum.shape:
            avg_sum += trigger
    return avg_sum/float(len(spikes))

def stc(stim,spikes,filter_shape=(10,1,1),binary=True,threshold=0.5):
    """Calculates Spike Triggered Covariance

        Parameters
        ----------
    
        stim (np.array(time,x,y)):
            stimulus presented
        spikes (np.array(time,x,y)):
            spikes recorded
        filter_shape (tuple)
            The desired output filter shape
        binary (bool)
            Whether the spikes are already a binary sequence (0s and 1s or True and False)
        threshold (float)
            If binary is False, this threshold converts the input array into
            a binary array

        Notes
        -----
        For now we ignore spikes too close to the edge!
    """
    if binary:
        spikes = np.array(np.where(spikes)).transpose()
    else:
        spikes = np.array(np.where(spikes>=threshold)).transpose()
    avg_sum = np.zeros((np.prod(filter_shape),np.prod(filter_shape)))
    avg_count = len(spikes)
    for s in spikes:
        trigger = stim[s[0]:s[0]+filter_shape[0],
                        s[1]:s[1]+filter_shape[1],
                        s[2]:s[2]+filter_shape[2]]
        if trigger.shape == filter_shape:
            avg_sum += trigger.flatten()[None,:]*trigger.flatten()[:,None]
    return avg_sum/float(len(spikes))


class LinearGradientExplorer(object):
    """A class to explore the error of a model over a linear extension of a gradient.

    Example::

        >> m = convis.models.LN()
        >> le = LinearGradientExplorer(m,inp,outp,['conv_weight'])
        >> le.get_loss_at(0.0) # at the initial parameter values

        >> le.get_loss_at(np.linspace(-10.0,10.0,100)) # at 100 values from -10x the gradient to 10x the gradient

        >> le.get_loss_at(np.linspace(-10.0,10.0,100),grad_override={'conv_weight':1.0}) # overriding the gradient with 1.0

    """
    def __init__(self,model,inp,outp,dt=100,vars=None,loss_func=None):
        """

        """
        self.model = model
        self.inp = inp
        self.outp = outp
        self.dt = dt
        self.loss_func = loss_func
        if self.loss_func is None:
            self.loss_func = lambda x,y: torch.mean((x-y)**2.0)
        self.vars = vars
        if self.vars is None:
            self.vars = self.model.p._all.__iterkeys__()
        self.p0 = self.get_params()
        self.compute_loss()
        self.g0 = self.get_grads()
    def get_loss_at(self,k,grad_override={}):
        if type(k) in [list,tuple,np.ndarray]:
            return np.array([self.get_loss_at(kk,grad_override=grad_override) for kk in k])
        else:
            g = copy.copy(self.g0)
            g.update(grad_override)
            self.set_params(self.p0,k,g)
            return self.compute_loss()[0].data[0]
    def compute_loss(self):
        try:
            self.zero_grads()
        except:
            pass
        self.model.clear_state()
        return self.model.compute_loss(self.inp,self.outp,dt=self.dt,loss_fn=self.loss_func)
    def zero_grads(self):
        for k,p in self.model.p._all.__iteritems__():
            p.grad.data.zero_()
    def get_grads(self):
        grads = {}
        for k,p in self.model.p._all.__iteritems__():
            if k not in self.vars:
                continue
            grads[k] = p.grad.data.numpy().copy()
        return grads
    def get_params(self):
        params = {}
        for k,p in self.model.p._all.__iteritems__():
            if k not in self.vars:
                continue
            params[k] = p.data.numpy().copy()
        return params
    def set_params(self,p0,d,g0):
        for k,p in self.model.p._all.__iteritems__():
            if k not in self.vars:
                continue
            if hasattr(p,'set'):
                p.set(p0[k]+d*g0[k])
            else:
                p.data = torch.Tensor(p0[k]+d*g0[k])