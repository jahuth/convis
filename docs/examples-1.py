import convis
import numpy as np
some_input = convis.samples.moving_grating()
spk = convis.filters.spiking.LeakyIntegrateAndFireNeuron()
spk.p.g_L = 1.0
o = spk.run(20.0*some_input,dt=100)
plt.figure()
o.plot(mode='lines')