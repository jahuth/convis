import convis
inp = convis.samples.moving_grating(2000,50,50)
convis.utils.plot_tensor(inp[None,None],n_examples=10,max_lines=42)