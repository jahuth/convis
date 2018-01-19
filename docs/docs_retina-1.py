import convis
import numpy as np
from matplotlib import pylab as plt
retina = convis.retina.Retina()
inp = convis.samples.moving_grating(2000)
inp = np.concatenate([inp[:1000],2.0*inp[1000:1500],inp[1500:2000]],axis=0)
o_init = retina.run(inp[:500],dt=200)
o = retina.run(inp[500:],dt=200)
convis.plot_5d_time(o[0],mean=(3,4)) # plots the mean activity over time
plt.figure()
retina = convis.retina.Retina(opl=True,bipolar=False,gang=True,spikes=False)
o_init = retina.run(inp[:500],dt=200)
o = retina.run(inp[500:],dt=200)
convis.plot_5d_time(o[0],mean=(3,4)) # plots the mean activity over time
convis.plot_5d_time(o[0],alpha=0.1) # plots a line for each pixel