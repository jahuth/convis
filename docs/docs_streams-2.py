import convis
inp = convis.streams.RandomStream((20,20),level=3.0,mean=10.0)
convis.plot(inp.get(100))