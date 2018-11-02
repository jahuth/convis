import convis
import numpy as np
import matplotlib.pylab as plt
T,X,Y = np.meshgrid(np.linspace(0.0,10.0,1000),
                np.linspace(-2.0,2.0,20),
                np.linspace(-2.0,2.0,20), indexing='ij')
some_input = np.sin(T+X+Y)
plt.matshow(some_input[3,:,:])
plt.show()
spk = convis.filters.spiking.LeakyIntegrateAndFireNeuron()
spk.p.g_L = 10.0
o = spk.run(some_input,dt=100)
plt.figure()
o.plot(mode='lines')
plt.show()