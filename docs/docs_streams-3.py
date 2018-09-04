import convis
x = np.ones((200,50,50))
x[:,10,:] = 0.0 
x[:,20,:] = 0.0 
x[:,30,:] = 0.0 
x *= np.sin(np.linspace(0.0,12.0,x.shape[0]))[:,None,None]
x += np.sin(np.linspace(0.0,12.0,x.shape[1]))[None,:,None]
inp = convis.streams.SequenceStream(x)
convis.plot(inp.get(100))