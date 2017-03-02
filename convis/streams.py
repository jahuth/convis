import numpy as np
import glob,litus

"""
Test::

    %pylab inline
    import litus
    import convis.runner
    import convis.streams
    A = np.array([1.0*(np.arange(50*50) < i).reshape((50,50)) for i in range(50*50 + 1)])
    inp = convis.streams.RepeatingStream(A,size=(50,50))
    out = convis.streams.run_visualizer()
    out.refresh_rate = 5
    f = lambda x: [np.sqrt(x)]
    r = convis.runner.Runner(f,inp,out,dt=0.005)

"""

class GlobalClock(object):
    def __init__(self, t_zero = 0.0, t=0.0, dt=0.001):
        self.t = t
        self.t_zero = t_zero
        self.dt = dt
    def offset(self,d):
        return lambda: self.t + d


class InrImageStreamer(object):
    """ Reads a large inr file and is an iterator over all images.

        By default, the z dimension is ignored. Setting `z` to True changes this, such that each image is 3d instead of 2d.
    """
    def __init__(self,filename,z=False, slice_at = None):
        self.filename = filename
        self.file = open(filename,'r')
        self.raw_header = self.file.read(256)
        self.header = dict([h.split('=') for h in self.raw_header.split('\n') if '=' in h])
        self.header = dict([(k,litus._make_an_int_if_possible(v)) for k,v in self.header.iteritems()])
        self.z = z
        self.start_at = 0
        self.stop_at = None
        self.step_at = None
        if slice_at is not None:
            self.start_at = slice_at.start
            self.stop_at = slice_at.stop
            self.step_at = slice_at.step
        if self.start_at is None:
            self.start_at = 0
        if self.step_at is None:
            self.step_at = 1
        self.image_i = self.start_at
    def reset(self):
        self.image_i = self.start_at
    def skip(self,i=1):
        self.image_i += i
        self.file.skip(self.header['VDIM']*self.header['XDIM']*self.header['YDIM']*8)
    def seek(self,i):
        self.image_i = i
        if i >= self.header['ZDIM']:
            raise StopIteration()
        self.file.seek(256 + i * self.header['VDIM']*self.header['XDIM']*self.header['YDIM']*8)
    def read(self,i):
        self.seek(i)
        if self.header['TYPE'] == 'float' and self.header['PIXSIZE'] == '64 bits':
            if self.z:
                return np.array([[np.frombuffer(self.file.read(self.header['VDIM']*8),'float') for y in range(self.header['YDIM'])] for x in range(self.header['XDIM'])])
            else:
                return np.array([np.concatenate([np.frombuffer(self.file.read(self.header['VDIM']*8),'float') for y in range(self.header['YDIM'])],0) for x in range(self.header['XDIM'])])
        else:
            raise Exception("Not Implemented. So far only 8 byte floats are supported")
    def _read_one_image(self):
        self.image_i += 1
        if self.header['TYPE'] == 'float' and self.header['PIXSIZE'] == '64 bits':
            return np.array([[np.frombuffer(self.file.read(self.header['VDIM']*8),'float') for y in range(self.header['YDIM'])] for x in range(self.header['XDIM'])])
        else:
            raise Exception("Not Implemented. So far only 8 byte floats are supported")
    def __iter__(self):
        def f():
            image_i = self.start_at
            max_image = self.header['ZDIM']
            if self.stop_at is not None:
                max_image = self.stop_at
            while image_i < max_image:
                yield self.read(image_i)
                image_i += self.step_at
            raise StopIteration()
        return f()
    def __getitem__(self,indx):
        if type(indx) is int:
            return self.read(indx)
        if type(indx) is slice:
            return ImageStreamer(self.filename,z=self.z, slice_at=indx)
    def __len__(self):
        image_i = self.start_at
        max_image = self.header['ZDIM']
        if self.stop_at is not None:
            max_image = self.stop_at
        return max_image - image_i
    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.file.close()

class InrImageStreamWriter(object):
    def __init__(self,filename,v=0,x=0,y=0,z=0):
        self.filename = filename
        self.file = open(filename,'w')
        self.image_i = 0
        self.header = {'CPU': 'decm',
                         'PIXSIZE': '64 bits',
                         'TYPE': 'float',
                         'VDIM': v,
                         'XDIM': x,
                         'YDIM': y,
                         'ZDIM': z}
        #self.write_header()
    def write_header(self):
        self.file.seek(0)
        self.header['VDIM'] = self.image_i
        header = "#INRIMAGE-4#{\n"+"\n".join([str(k) +'='+str(v) for k,v in self.header.iteritems()])
        header = header + ('\n' * (252 - len(header))) + "##}"
        self.file.write(header)
    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.write_header()
        self.file.close()
    def seek(self,i):
        self.file.seek(255 + i * self.header['ZDIM']*self.header['XDIM']*self.header['YDIM']*8)
    def write(self,img):
        if self.header['TYPE'] == 'float' and self.header['PIXSIZE'] == '64 bits':
            buff = np.array(img,dtype='float64').tobytes()
            if len(buff) == self.header['ZDIM']*self.header['XDIM']*self.header['YDIM']*8:
                self.seek(self.image_i)
                self.file.write(buff)
                self.image_i += 1
            else:
                raise Exception("Image has incorrect size:"+str(len(buff))+" != "+str(self.header['ZDIM']*self.header['XDIM']*self.header['YDIM']*8))
        else:
            raise Exception("Not Implemented. So far only 8 byte floats are supported")

class InrImageFileStreamer(object):
    def __init__(self,filenames):
        if type(filenames) is str:
            filenames = litus.natural_sorted(glob.glob(filenames))
        self.filenames = filenames
        self.lengths = []
        self.file_i = 0
        self.fileobject = None
        self.fileobject_index = None
        for i,f in enumerate(self.filenames):
            with ImageStreamer(f) as ims:
                self.lengths.append(len(ims))
    def reset(self):
        self.file_i = 0
        self.image_i = 0
    def __iter__(self):
        def f():
            file_i = 0
            image_i = 0
            while file_i < len(self.filenames):
                ims = ImageStreamer(self.filenames[file_i])
                image_i = 0
                try:
                    while True:
                        yield ims.read(image_i)
                        image_i += 1
                except StopIteration:
                    file_i+=1
            raise StopIteration()
        return f()
    def skip(self,f=0,i=0):
        self.file_i += f
        self.image_i += i
        self.seek(self.file_i, self.image_i)
    def seek(self,f,i):
        self.file_i = f
        self.image_i = i
        while self.image_i >= self.lengths[self.file_i]:
            self.image_i -= self.lengths[self.file_i]
            self.file_i += 1
            if self.file_i >= len(self.filenames):
                raise StopIteration()
    def read(self):
        if self.fileobject_index != self.file_i:
            self.fileobject = ImageStreamer(self.filenames[self.file_i])
            self.fileobject_index = self.file_i
        return self.fileobject.read(self.image_i)
    def __len__(self):
        return np.sum(self.lengths)
    def __getitem__(self,indx):
        if type(indx) is int:
            self.seek(f=0,i=indx)
            return self.read()

class Stream(object):
    """Basic stream that gives zeros"""
    def __init__(self, size=(50,50), pixel_per_degree=10, t_zero = 0.0, t=0.0, dt=0.001):
        self.size = list(size)
        self.pixel_per_degree = pixel_per_degree
        self.t = t
        self.t_zero = t_zero
        self.dt = dt
    def time_to_bin(self,t):
        return (t-self.t_zero)/self.dt
    def __iter__(self):
        while True:
            self.t += self.dt
            yield np.zeros(self.size)
    def available(self,l=1):
        return True
    def get(self,i):
        self.t += i * self.dt
        return np.zeros([i]+self.size)
    def put(self,s):
        raise Exception("Not implemented for basic stream.")

class RandomStream(Stream):
    def __init__(self, size=(50,50), pixel_per_degree=10, level=1.0):
        self.level = level
        self.size = list(size)
        self.pixel_per_degree = pixel_per_degree
    def __iter__(self):
        while True:
            yield self.level * np.random.rand(*self.size)
    def available(self,l=1):
        return True
    def get(self,i):
        return self.level * np.random.rand(*([i]+self.size))
    def put(self,s):
        raise Exception("Not implemented for basic stream.")

class SequenceStream(Stream):
    """ 3d Numpy array that represents a sequence of images"""
    def __init__(self, sequence=np.zeros((0,50,50)), size=None, pixel_per_degree=10):
        self.size = sequence.shape[1:]
        self.pixel_per_degree = pixel_per_degree
        self.sequence = sequence
        self.i = 0
    def __iter__(self):
        while len(self.sequence) < self.i:
            self.i += 1
            yield self.sequence[i-1]
    def available(self,l=1):
        return (len(self.sequence) - self.i) > l
    def get(self,i):
        self.i += i
        return self.sequence[(self.i-i):self.i]
    def put(self,s):
        self.sequence = np.concatenate([self.sequence,s],axis=0.0)

class RepeatingStream(Stream):
    def __init__(self, sequence=np.zeros((0,50,50)), size=None, pixel_per_degree=10):
        self.size = sequence.shape[1:]
        self.pixel_per_degree = pixel_per_degree
        self.sequence = sequence
        self.i = 0
    def __iter__(self):
        while True:
            self.i += 1
            while len(self.sequence) < self.i:
                self.i=0
            yield self.sequence[i-1]
    def available(self,l=1):
        return len(sequence) > 0
    def get(self,i):
        self.i += i
        if len(self.sequence) < self.i:
            pre_index = self.i-i
            self.i -= len(self.sequence)
            return np.concatenate([self.sequence[pre_index:],self.sequence[:self.i]],axis=0)
        return self.sequence[(self.i-i):self.i]
    def put(self,s):
        self.sequence = np.concatenate([self.sequence,s],axis=0.0)


class TimedSequenceStream(object):
    """ 3d Numpy array that represents a sequence of images"""
    def __init__(self, sequence=np.zeros((0,50,50)), size=None, pixel_per_degree=10, t_zero = 0.0, dt = 0.001):
        self.size = sequence.shape[1:]
        self.pixel_per_degree = pixel_per_degree
        self.sequence = sequence
        self.t_zero = t_zero
        self.dt = dt
    def time_to_bin(self,t1,t2):
        return (np.round((t1-self.t_zero)/self.dt),
                         np.round((t2-self.t_zero)/self.dt))
    def xs(self,t1,t2):
        return np.arange(np.round((t1-self.t_zero)/self.dt),
                         np.round((t2-self.t_zero)/self.dt))
    def ts(self,t1,t2):
        return self.dt*np.arange(np.round((t1-self.t_zero)/self.dt),
                         np.round((t2-self.t_zero)/self.dt),1.0)
    def available(self,t1,t2):
        b1,b2 = self.time_to_bin(t1,t2)
        return b1 >= 0 and b2 <= len(self.sequence)
    def get(self,t1,t2):
        b1,b2 = self.time_to_bin(t1,t2)
        return self.sequence[b1:b2]
    def get_tsvs(self,t1,t2):
        b1,b2 = self.time_to_bin(t1,t2)
        if b1 < 0:
            b1 = 0
        if b2 > len(self.sequence):
            b2 = len(self.sequence)
        ts = self.dt*np.arange(b1,b2,1.0)
        return ts,self.sequence[b1:b2]
    def put(self,t1,t2, s):
        b1,b2 = self.time_to_bin(t1,t2)
        self.sequence[b1:b2] = s
        
        
class TimedResampleStream(TimedSequenceStream):
    """ 3d Numpy array that represents a sequence of images"""
    def __init__(self, stream, t_zero = 0.0, dt = 0.001):
        self.stream = stream
        self.t_zero = t_zero
        self.dt = dt
    def time_to_bin(self,t1,t2):
        return (np.round((t1-self.t_zero)/self.dt),
                         np.round((t2-self.t_zero)/self.dt))
    def xs(self,t1,t2):
        return np.arange(np.round((t1-self.t_zero)/self.dt),
                         np.round((t2-self.t_zero)/self.dt))
    def available(self,t1,t2):
        return self.stream.available(t1,t2)
    def get_tsvs(self,t1,t2):
        return self.stream.get_tsvs(t1,t2)
    def get(self,t1,t2):
        t,v = self.stream.get_tsvs(t1-2.0*self.dt,t2+2.0*self.dt)
        try:
            return interp1d(t,
                        v.reshape(len(t),-1),
                        axis=0,
                        fill_value='extrapolate',
                        bounds_error = False
                       )(self.ts(t1,t2)).reshape(
            [len(self.ts(t1,t2))]+list(v.shape[1:]))
        except ValueError:
            # old versions of scipy don't know extrapolate
            # it also doesn't behave as numpy interpolate (extending the first and last values) as only one value is accepted
            # this should not be a problem if we 2*dt before and after the time slice
            return interp1d(t,
                        v.reshape(len(t),-1),
                        axis=0,
                        fill_value=np.mean(v),
                        bounds_error = False
                       )(self.ts(t1,t2)).reshape(
            [len(self.ts(t1,t2))]+list(v.shape[1:]))
    def put(self,t1,t2, s):
        self.stream.put(t1,t2)

_started_tkinter_main_loop = False
_main_root = None

class StreamVisualizer():
    def __init__(self):
        self.dirty = False
        self.refresh_rate = 1
        self.last_buffer = []
        self.closed = False
        #self.recieve_thread = thread.start_new_thread(self.recieve,tuple())
    def mainloop(self):
        import Tkinter as tk
        from PIL import Image, ImageTk
        from ttk import Frame, Button, Style
        import time
        import cStringIO
        import socket
        self.root = tk.Toplevel() #Tk()
        self.root.title('Display')
        self.buffer = cStringIO.StringIO()
        self.image = Image.fromarray(np.zeros((200,200))).convert('RGB')
        self.image1 = ImageTk.PhotoImage(self.image)
        self.panel1 = tk.Label(self.root, image=self.image1)
        self.display = self.image1
        self.panel1.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.root.after(100, self.update_image)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        #global _started_tkinter_main_loop
        #if not _started_tkinter_main_loop:
        #    _started_tkinter_main_loop = True
        #    print "Starting Tk main thread..."

    def on_closing(self):
        self.closed = True
        self.root.destroy()
    def put(self,img):
        if self.closed:
            pass #
        self.last_buffer.extend([i for i in img])
    def update_image(self):
        from PIL import Image, ImageTk
        refresh = int(self.refresh_rate)
        if len(self.last_buffer) > 0:
            try:
                image_buffer = self.last_buffer.pop(0) # take image from the front
                self.image = Image.fromarray(256.0*image_buffer).convert('RGB')#Image.open(image_buffer)#cStringIO.StringIO(self.last_buffer))
                self.image.load()
                self.image1 = ImageTk.PhotoImage(self.image)
                self.panel1.configure(image=self.image1)
                self.root.title(str(len(self.last_buffer))+' Images buffered')
                self.display = self.image1
                refresh = self.image.info.get('refresh',refresh)
            except Exception as e:
                #print e
                raise
                #pass
        self.root.after(refresh, self.update_image)

def _create_mainloop():
    import Tkinter as tk
    global _main_root
    _main_root = tk.Tk()
    _main_root.title('Hidden Display')
    _main_root.withdraw()
    _started_tkinter_main_loop = True
    print "Starting Tk main thread..."
    _main_root.mainloop()    

def run_visualizer():
    import thread, time
    global _main_root
    if _main_root is None:
        thread.start_new_thread(_create_mainloop,tuple())
        time.sleep(0.1)
    s = StreamVisualizer()
    thread.start_new_thread(s.mainloop,tuple())
    return s

class StreamToNetwork(Stream):
    def __init__(self,port,host='localhost',compress_level=0,resize=(1.0,1.0)):
        self.port = port
        self.host = host
        self.compress_level=compress_level
        self.resize = resize
    def put(self,s):
        from . import png
        png.png_client(s,info={},port=self.port,host=self.host,compress_level=self.compress_level,resize=self.resize)
