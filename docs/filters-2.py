import convis
import numpy as np
import matplotlib.pylab as plt
m = convis.filters.RF()
inp = convis.samples.moving_grating()
o = m.run(inp, dt=200)
o.plot(label='uniform rf')
m.set_weight(np.random.randn(*m.weight.size()))
o = m.run(inp, dt=200)
o.plot(label='random rf')
plt.legend()