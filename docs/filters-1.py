import convis
import numpy as np
import matplotlib.pylab as plt
plt.figure()
plt.imshow(convis.numerical_filters.gauss_filter_2d(4.0,4.0))
plt.show()