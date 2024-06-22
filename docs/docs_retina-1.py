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

#add 6.22 input test
import numpy as np
import matplotlib.pyplot as plt

# Creates an array of shape (2000, 20, 20) with all elements initialized to 1
input_data = np.ones((2000, 20, 20))

# View the shape of the array
print("Array shape:", input_data.shape)

# View the first five elements of the first time step (frame 1) of the array
print("The first five elements of the first time step (frame 1):\n", input_data[0, :5, :5])

# View basic array information
print("The data type of the array:", input_data.dtype)
print("Array dimension:", input_data.ndim)
print("Size of array:", input_data.size)

# Visual first frame
plt.imshow(input_data[0], cmap='gray')
plt.title('Frame 1')
plt.colorbar()
plt.show()
