"""
Analysis Tools
--------------



"""
import numpy as np

__all__ = ['sta','stc']

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