import convis
from matplotlib.pylab import plot, xlim, gcf
stream = convis.streams.PoissonMNISTStream('../data',rep=20,fr=0.20) 
# here we are using a very high firing rate for easy visualization
# (20% of cells are active in each frame)
convis.plot(stream.get(500))
gcf().show()
convis.plot(stream.get(500))
gcf().show()
stream.reset() # returning to the start
convis.plot(stream.get(500))
gcf().show()