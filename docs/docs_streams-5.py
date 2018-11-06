import convis
from matplotlib.pylab import plot, xlim, gcf
stream = convis.streams.MNISTStream('../data',rep=20)
convis.plot(stream.get(500))
gcf().show()
convis.plot(stream.get(500))
gcf().show()
stream.reset() # returning to the start
convis.plot(stream.get(500))
gcf().show()