

class Stream(object):
    def __init__(self, size=(50,50), pixel_per_degree=10):
        self.size = list(size)
        self.pixel_per_degree = pixel_per_degree
    def __iter__(self):
        return np.zeros(self.size)
    def available(self):
        return -1
    def get(self,i):
        return np.zeros([i]+self.size)

class RandomStream(Stream):
    def __init__(self, size=(50,50), pixel_per_degree=10, level=1.0):
        self.level = level
        self.size = list(size)
        self.pixel_per_degree = pixel_per_degree
    def __iter__(self):
        return self.level * np.random.rand(*self.size)
    def available(self):
        return -1
    def get(self,i):
        return self.level * np.random.rand(*([i]+self.size))

class SequenceStream(Stream):
    def __init__(self, sequence=np.zeros((0,50,50)), size=None, pixel_per_degree=10):
        self.size = sequence.shape[1:]
        self.pixel_per_degree = pixel_per_degree
        self.sequence = sequence
        self.i = 0
    def __iter__(self):
        self.i += 1
        if len(self.sequence) < self.i:
            raise Exception('End of Sequence reached!')
        return self.sequence[i-1]
    def available(self):
        return len(self.sequence) - self.i
    def get(self,i):
        self.i += i
        return self.sequence[(self.i-i):self.i]

class RepeatingStream(Stream):
    def __init__(self, sequence=np.zeros((0,50,50)), size=None, pixel_per_degree=10):
        self.size = sequence.shape[1:]
        self.pixel_per_degree = pixel_per_degree
        self.sequence = sequence
        self.i = 0
    def __iter__(self):
        self.i += 1
        if len(self.sequence) < self.i:
            self.i=0
        return self.sequence[i-1]
    def available(self):
        return -1
    def get(self,i):
        self.i += i
        if len(self.sequence) < self.i:
            pre_index = self.i-i
            self.i -= len(self.sequence)
            return np.concatenate([self.sequence[pre_index:],self.sequence[:self.i]],axis=0)
        return self.sequence[(self.i-i):self.i]
