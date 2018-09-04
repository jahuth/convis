import convis
inp = convis.samples.moving_grating(2000,50,50)
inp = np.concatenate([-1.0*inp[None,None],2.0*inp[None,None],inp[None,None]],axis=1)
convis.utils.plot_tensor_with_channels(inp[None,None],n_examples=2,max_lines=2)