import convis
model = convis.models.Retina()
inp = convis.streams.RandomStream((20,20),mean=127,level=100.0)
o = model.run(inp,dt=100, max_t=1000)
convis.plot(o[0])