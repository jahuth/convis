import convis
inp = convis.samples.moving_grating(2000,50,50)
convis.utils.plot_5d_matshow(inp[None,None,::200])