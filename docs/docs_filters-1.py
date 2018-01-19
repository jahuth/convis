import matplotlib.pyplot as plt
import numpy as np
import convis
s = convis.filters.SmoothConv(n=6,tau=0.05)
inp = np.zeros((1000,1,1))
inp[50,0,0] = 1.0
inp = convis.prepare_input(inp)
c = s.get_all_components(inp)
convis.plot_5d_time(c,mean=(3,4))
c = c.data.cpu().numpy()