.. _filters:
Layers
========


.. plot::

    import convis
    import numpy as np
    import matplotlib.pylab as plt
    plt.figure()
    plt.imshow(convis.numerical_filters.gauss_filter_2d(4.0,4.0))
    plt.show()


.. plot::

    import convis
    import numpy as np
    import matplotlib.pylab as plt
    plt.figure()
    plt.plot(convis.numerical_filters.exponential_filter_1d(tau=0.01))
    plt.show()

