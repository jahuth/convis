import numpy as np
import matplotlib.pylab as plt

import convis
v = 10.0
the_input = np.concatenate([contrast * convis.samples.moving_bars(t=200, vt=20, vx=v*np.sin(phi), vy=v*np.cos(phi))
                            for phi in np.linspace(0,360.0,30.0) for contrast in [0.0,1.0]], axis = 0)

receptors = convis.filters.simple.ConvolutionFilter2d(
    {'kernel': convis.numerical_filters.gauss_filter_2d(2.0,2.0) },
    name='ReceptorLayer')
horizontal_cells = convis.filters.simple.ConvolutionFilter2d(
    {'kernel': convis.numerical_filters.gauss_filter_2d(4.0,4.0) },
    name='HorizontalLayer')
rf = convis.RF_2d_kernel_filter(
    {'kernel': convis.samples.gabor_kernel(size=the_input.shape[1]) },
    name='ReceptiveField')


horizontal_cells += receptors
rf += receptors
rf += -0.5*horizontal_cells.graph

m = convis.make_model(rf)
o = m.run(the_input)
plt.plot(o[0][:1000,10,:],alpha=0.5)
plt.xlabel('time (eg. ms)')

plt.plot(o[0][:,:,:].clip(0,None).mean((1,2)),alpha=0.5)
plt.show()