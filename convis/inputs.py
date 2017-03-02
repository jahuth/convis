import png

class Input(object):
    def __init__(self, size=(50,50), pixel_per_degree=10):
        self.size = list(size)
        self.pixel_per_degree = pixel_per_degree
    def __iter__(self):
        return np.zeros(self.size)
    def available(self):
        return -1
    def get(self,i):
        return np.zeros([i]+self.size)


class NetworkInput(Input):
    def __init__(self, port=10000, size=(50,50), pixel_per_degree=10):
        self.size = list(size)
        self.pixel_per_degree = pixel_per_degree
        self.server = png.ImageRecieverServer(port,size=self.size)
    def __iter__(self):
        while self.available():
            yield self.server.get()[0]
    def available(self):
        return self.server.available()
    def get(self,i):
        return self.server.get(i)
    def __len__(self):
        return len(self.server)
