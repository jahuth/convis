"""
The streams module contains `Stream` classes that can load different kinds of data.

Streams that are usable currently are: :class:`RandomStream`, :class:`SequenceStream` and :class:`ImageSequence`.
All others are experimental. If you plan on using any of them, you can open a  `enhancement <https://github.com/jahuth/convis/labels/enhancement>`_ issue on the issue tracker.


Since some streams do not have a defined end, the argument `max_t` can specify for how long a model should be executed.

.. plot ::
    :include-source:

    import convis
    model = convis.models.Retina()
    inp = convis.streams.RandomStream((20,20),mean=127,level=100.0)
    o = model.run(inp,dt=100, max_t=1000)
    convis.plot(o[0])


"""

from __future__ import print_function
import numpy as np
import glob,litus
import datetime
from future.utils import iteritems as _iteritems


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

    .. warning::

        This class is not currently maintained and might change without notice between releases.

    """
    def __init__(self,filename,z=False, slice_at = None):
        self.filename = filename
        self.file = open(filename,'r')
        self.raw_header = self.file.read(256)
        self.header = dict([h.split('=') for h in self.raw_header.split('\n') if '=' in h])
        self.header = dict([(k,litus._make_an_int_if_possible(v)) for k,v in _iteritems(self.header)])
        self.z = z
        self.last_image = np.zeros((50,50))
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
    @property
    def shape(self):
        return (self.header['VDIM'],self.header['XDIM'],self.header['YDIM'])
    def reset(self):
        self.image_i = self.start_at
    def skip(self,i=1):
        self.image_i += i
        self.file.skip(self.header['ZDIM']*self.header['XDIM']*self.header['YDIM']*8)
    def seek(self,i):
        self.image_i = i
        if i >= self.header['VDIM']:
            raise StopIteration()
        self.file.seek(256 + i * self.header['ZDIM']*self.header['XDIM']*self.header['YDIM']*8)
    def read(self,i):
        self.seek(i)
        if self.header['TYPE'] == 'float' and self.header['PIXSIZE'] == '64 bits':
            if self.z:
                self.last_image = np.array([[np.frombuffer(self.file.read(self.header['ZDIM']*8),'float') for y in range(self.header['YDIM'])] for x in range(self.header['XDIM'])])
            else:
                self.last_image = np.array([np.concatenate([np.frombuffer(self.file.read(self.header['ZDIM']*8),'float') for y in range(self.header['YDIM'])],0) for x in range(self.header['XDIM'])])
            return self.last_image
        else:
            raise Exception("Not Implemented. So far only 8 byte floats are supported")
    def get_image(self):
        return self.last_image
    def _read_one_image(self):
        self.image_i += 1
        if self.header['TYPE'] == 'float' and self.header['PIXSIZE'] == '64 bits':
            a = np.array([[np.frombuffer(self.file.read(self.header['ZDIM']*8),'float') for y in range(self.header['YDIM'])] for x in range(self.header['XDIM'])])
            return a[:,:,0]
        else:
            raise Exception("Not Implemented. So far only 8 byte floats are supported")
    def get(self,i=1):
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        return np.array([self._read_one_image() for ii in range(i)])
    def __iter__(self):
        def f():
            image_i = self.start_at
            max_image = self.header['VDIM']
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
        max_image = self.header['VDIM']
        if self.stop_at is not None:
            max_image = self.stop_at
        return max_image - image_i
    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.file.close()
    @property
    def buffered(self):
        return 0

class InrImageStreamWriter(object):
    """
    .. warning::

        This class is not currently maintained and might change without notice between releases.

    Example
    -------

        s = convis.streams.InrImageStreamWriter('stim.inr',x=inp.shape[1],y=inp.shape[2],time_dimension='ZDIM')
        with s:
            for i in inp:
                s.put(i[None,:,:])

    """
    def __init__(self,filename,v=1,x=1,y=1,z=1,time_dimension='VDIM'):
        """
            Usage::

                s = convis.streams.InrImageStreamWriter('stim.inr',x=inp.shape[1],y=inp.shape[2],time_dimension='ZDIM')
                with s:
                    for i in inp:
                        s.put(i[None,:,:])
        """
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
        self.last_image = np.zeros((50,50))
        self.time_dimension = time_dimension
        #self.write_header()
    def write_header(self):
        self.file.seek(0)
        self.header[self.time_dimension] = self.image_i
        header = "#INRIMAGE-4#{\n"+"\n".join([str(k) +'='+str(v) for k,v in _iteritems(self.header)])
        header = header + ('\n' * (252 - len(header))) + "##}"
        self.file.write(header)
    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.write_header()
        self.file.close()
    def seek(self,i):
        self.file.seek(255 + i * self.header['ZDIM']*self.header['XDIM']*self.header['YDIM']*8)
    def get_image(self):
        return self.last_image
    def put(self,seq):
        for s in seq:
            self.write(s)
            self.last_image = s
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
    def _web_repr_(self,name,namespace):
        s= "Inr Stream Writer Object: <br>"
        s += " + file: "+str(self.filename)+" <br>"
        s += " + length: "+str(self.image_i)+" <br>"
        if hasattr(self,'get_image'):
            s+= '<img class="mjpg" src="/mjpg/'+name+'" alt="'+name+'" height="400"/>'
        return s
    @property
    def buffered(self):
        return 0

class InrImageFileStreamer(object):
    """
    .. warning::

        This class is not currently maintained and might change without notice between releases.

    """
    def __init__(self,filenames):
        if type(filenames) is str:
            filenames = litus.natural_sorted(glob.glob(filenames))
        self.filenames = filenames
        self.lengths = []
        self.file_i = 0
        self.fileobject = None
        self.fileobject_index = None
        for i,f in enumerate(self.filenames):
            with InrImageStreamer(f) as ims:
                self.lengths.append(len(ims))
    def reset(self):
        self.file_i = 0
        self.image_i = 0
    def __iter__(self):
        def f():
            file_i = 0
            image_i = 0
            while file_i < len(self.filenames):
                ims = InrImageStreamer(self.filenames[file_i])
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
            self.fileobject = InrImageStreamer(self.filenames[self.file_i])
            self.fileobject_index = self.file_i
        return self.fileobject.read(self.image_i)
    def get_image(self):
        return self.fileobject.get_image()
    def __len__(self):
        return np.sum(self.lengths)
    def __getitem__(self,indx):
        if type(indx) is int:
            self.seek(f=0,i=indx)
            return self.read()
    def _web_repr_(self,name,namespace):
        s= "Inr Stream Object:<br>"
        s += " + file: "+str(self.filename)+" <br>"
        s += " + length: "+str(self.image_i)+" <br>"
        if hasattr(self,'get_image'):
            s+= '<img class="mjpg" src="/mjpg/'+name+'" alt="'+name+'" height="400"/>'
        return s
    @property
    def buffered(self):
        return 0


class Stream(object):
    """
        Stream Base Class

        Streams have to have methods to either `get(i)`
        or `put` a frame.

         - `get(i)` takes `i` frames from the stream and changes the internal state.
         - `put(t)` inserts a `torch.Tensor` or `numpy.ndarray` into the stream (eg. saving it into a file).

    """
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
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        self.t += i * self.dt
        return np.zeros([i]+self.size)
    def put(self,s):
        raise Exception("Not implemented for basic stream.")
    def close(self):
        pass
    def _web_repr_(self,name,namespace):
        import cgi
        s= "Image Stream Object:<br>"
        s+= " + class: "+cgi.escape(str(self.__class__))+"<br>"
        try:
            s+= " + time: "+str(self.t)+"<br>"
        except:
            pass
        try:
            s+= " + length: "+str(len(self))+"<br>"
        except:
            s+= " + length: -- <br>"
        if hasattr(self,'get_image'):
            s+= '<img class="mjpg" src="/mjpg/'+name+'" alt="'+name+'" height="400"/>'
        return s
    @property
    def buffered(self):
        return 0

class ProcessingStream(Stream):
    """Combines a stream and a `Layer` (model or filter) into a new stream.
    
    Each slice will be processed at once (using `Layer.__call__`, not `Layer.run`), 
    so a limited number of frames should be requested at any one time.

    .. versionadded:: 0.6.4

    
    Parameters
    ----------
    input_stream : `convis.streams.Stream`
        The stream that is used as input to the model
    model : `convis.Layer`
        The model that transforms slices of the `input_stream` into outputs
    pre : function or None
        Optional operation to be done one each slice before handing it to the model.
    post : function or None
        Optional operation to be done one each slice after the model has processed it.

    Examples
    --------

    To recreate the neuromorphic MNIST stream one can combine the MNIST stream and a `Poisson` layer.
    The output of the MNIST stream can be scaled with a function passed to `pre`.

    .. code::

        import convis
        stream = convis.streams.MNISTStream('../data',rep=10)
        model = convis.filters.spiking.Poisson()
        stream2 = convis.streams.ProcessingStream(stream,model,pre=lambda x: 0.1*x)


    """
    def __init__(self, input_stream, model, pre = None, post = None):
        self.input_stream = input_stream
        self.model = model
        self.pre = pre
        if self.pre is None:
            self.pre = lambda x: x
        self.post = post
        if self.post is None:
            self.post = lambda x: x
    def get(self,i):
        return self.post(self.model(self.pre(self.input_stream.get(i))))

class RandomStream(Stream):
    """creates a stream of random frames.

    Examples
    --------

        .. plot::

            import convis
            inp = convis.streams.RandomStream((20,20),level=3.0,mean=10.0)
            convis.plot(inp.get(100))

    """
    def __init__(self, size=(50,50), pixel_per_degree=10, level=1.0, mean=0.0):
        self.level = level
        self.mean = mean
        self.size = list(size)
        self.pixel_per_degree = pixel_per_degree
    def __iter__(self):
        while True:
            yield self.level * np.random.rand(*self.size)
    def available(self,l=1):
        return True
    def get_image(self):
        return self.mean + self.level * np.random.rand(*(self.size))
    def get(self,i):
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        return self.mean + self.level * np.random.rand(*([i]+self.size))
    def put(self,s):
        raise Exception("Not implemented for read only stream.")

class SequenceStream(Stream):
    """ 3d Numpy array that represents a sequence of images

        .. warning::

            This class is not currently maintained and might change without notice between releases.

    Examples
    --------

        .. plot::
            :include-source:

            import convis
            x = np.ones((200,50,50))
            x[:,10,:] = 0.0 
            x[:,20,:] = 0.0 
            x[:,30,:] = 0.0 
            x *= np.sin(np.linspace(0.0,12.0,x.shape[0]))[:,None,None]
            x += np.sin(np.linspace(0.0,12.0,x.shape[1]))[None,:,None]
            inp = convis.streams.SequenceStream(x)
            convis.plot(inp.get(100))

    """
    def __init__(self, sequence=np.zeros((0,50,50)), size=None, pixel_per_degree=10, max_frames = 5000):
        self.size = sequence.shape[1:]
        self.pixel_per_degree = pixel_per_degree
        self.sequence = sequence
        self.i = 0
        self.max_frames = max_frames
    def __iter__(self):
        while len(self.sequence) < self.i:
            self.i += 1
            yield self.sequence[i-1]
    def __len__(self):
        return len(self.sequence)
    def available(self,l=1):
        return (len(self.sequence) - self.i) > l
    def get_image(self):
        try:
            if self.i < 1:
                return self.sequence[-1]
            return self.sequence[self.i]
        except:
            return np.zeros(self.sequence.shape[1:])
    def get(self,i):
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        self.i += i
        return self.sequence[int(self.i-i):self.i]
    def put(self,s):
        if self.sequence.shape[1:] == s.shape[1:]:
            if len(self.sequence) + len(s) > self.max_frames:
                self.sequence = np.concatenate([self.sequence,s],axis=0)[-self.max_frames:]
            else:
                self.sequence = np.concatenate([self.sequence,s],axis=0)
        else:
            if len(s) > self.max_frames:
                self.sequence = s[-self.max_frames:]
            else:
                self.sequence = s

class RepeatingStream(Stream):
    """
        .. warning::

            This class is not currently maintained and might change without notice between releases.
    """
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
    def get_image(self):
        try:
            return self.sequence[self.i]
        except:
            return np.zeros(self.sequence.shape[1:])
    def get(self,i):
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        self.i += i
        if len(self.sequence) < self.i:
            pre_index = int(self.i-i)
            self.i -= len(self.sequence)
            return np.concatenate([self.sequence[pre_index:],self.sequence[:self.i]],axis=0)
        return self.sequence[(self.i-i):self.i]
    def put(self,s):
        self.sequence = np.concatenate([self.sequence,s],axis=0.0)


class TimedSequenceStream(object):
    """ 3d Numpy array that represents a sequence of images

    .. warning::

        This class is not currently maintained and might change without notice between releases.
    """
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
    """ 3d Numpy array that represents a sequence of images

    .. warning::

        This class is not currently maintained and might change without notice between releases.

    """
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
        self.refresh_rate = 0.001
        self.minimal_refresh_rate = 0.01
        self.last_buffer = []
        self.closed = False
        self.auto_scale_refresh_rate = True
        self._last_put = datetime.datetime.now()
        self.decay = 0.05
        self.decay_activity = None
        self.last_batch_length = None
        self.last_batch_time = None
        self.cmap = None
        #self.recieve_thread = thread.start_new_thread(self.recieve,tuple())
    def mainloop(self):
        try:
            import Tkinter as tk
        except ImportError:
            import tkinter as tk
        from PIL import Image, ImageTk
        from ttk import Frame, Button, Style
        import time
        import socket
        self.root = tk.Toplevel() #Tk()
        self.root.title('Display')
        self.image = Image.fromarray(np.zeros((200,200))).convert('RGB')
        self.image1 = ImageTk.PhotoImage(self.image)
        self.panel1 = tk.Label(self.root, image=self.image1)
        self.display = self.image1
        self.frame1 = Frame(self.root, height=50, width=50)
        self.panel1.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.root.after(100, self.advance_image)
        self.root.after(100, self.update_image)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        #global _started_tkinter_main_loop
        #if not _started_tkinter_main_loop:
        #    _started_tkinter_main_loop = True
        #    print("Starting Tk main thread...")

    def on_closing(self):
        self.closed = True
        self.root.destroy()
    def put(self,img):
        if self.closed:
            pass #
        new_images = [i for i in img]
        self.last_buffer.extend(new_images)
        self.last_batch_length = len(new_images)
        if self.auto_scale_refresh_rate:
            self.last_batch_time = (datetime.datetime.now() - self._last_put).total_seconds()
            self.refresh_rate = self.last_batch_time/float(len(new_images))
            #print(self.refresh_rate)
        self._last_put = datetime.datetime.now()
        #if hasattr(self,'root'):
        #    self.root.title(str(len(self.last_buffer))+' Images buffered')
    def get_image(self):
        if self.decay_activity is not None:
            return self.decay_activity
        if len(self.last_buffer) > 0:
            return self.last_buffer[0]
        return np.zeros((50,50))
    def advance_image(self):
        from PIL import Image, ImageTk
        function_start = datetime.datetime.now()
        #refresh = (self.refresh_rate)
        half = len(self.last_buffer)*0.9
        num_frames = 0 # consume all
        #if self._last_put is not None and self.last_batch_length is not None and self.last_batch_time is not None:
        #    # only consume up to X images
        #    if self.last_batch_length > 0:
        #        num_frames = self.last_batch_length+((self._last_put - datetime.datetime.now()).total_seconds() + self.last_batch_time)/self.refresh_rate
        #while len(self.last_buffer) > 1.2*num_frames:
        for i in np.arange(np.floor(self.minimal_refresh_rate/self.refresh_rate)+1.0):
            if len(self.last_buffer) > 1:
                try:
                    image_buffer = self.last_buffer.pop(0) # take image from the front
                    if self.decay_activity is None:
                        self.decay_activity = 1.0*image_buffer
                    self.decay_activity -= self.decay * self.decay_activity
                    self.decay_activity += image_buffer
                    #refresh = self.image.info.get('refresh',refresh)
                except Exception as e:
                    #print(e)
                    raise
                    #pass
        function_lag = (datetime.datetime.now() - function_start).total_seconds()
        self.root.after(int(max((self.refresh_rate-function_lag),self.minimal_refresh_rate)*1000.0), self.advance_image)
    def update_image(self):
        from PIL import Image, ImageTk
        if self.decay_activity is not None:
            try:
                da = self.decay_activity
                da = da + np.min(da)
                im = da/max(np.max(da),1.0)
                im = im.clip(0.0,1.0)
                if im.shape[0] < 50:
                    im = np.repeat(im,10,axis=0)
                    im = np.repeat(im,10,axis=1)
                elif im.shape[0] < 100:
                    im = np.repeat(im,5,axis=0)
                    im = np.repeat(im,5,axis=1)
                elif im.shape[0] < 300:
                    im = np.repeat(im,2,axis=0)
                    im = np.repeat(im,2,axis=1)
                if self.cmap is not None:
                    self.image = Image.fromarray(self.cmap(im, bytes=True))
                else:
                    self.image = Image.fromarray(256.0*im).convert('RGB')#Image.open(image_buffer)#cStringIO.StringIO(self.last_buffer))
                #self.image.resize((500,500), Image.ANTIALIAS)
                #self.image.load()
                self.image1 = ImageTk.PhotoImage(self.image)
                self.panel1.configure(image=self.image1)
                self.root.title(str(len(self.last_buffer))+' Images buffered')
                self.display = self.image1
            except Exception as e:
                #print(e)
                raise
                #pass
        self.root.after(int(50), self.update_image)
    @property
    def buffered(self):
        return len(self.last_buffer)
    def _web_repr_(self,name,namespace):
        import cgi
        s= "Image Stream Object:<br>"
        s+= " + class: "+cgi.escape(str(self.__class__))+"<br>"
        try:
            s+= " + length: "+str(len(self))+"<br>"
        except:
            s+= " + length: -- <br>"
        if hasattr(self,'get_image'):
            s+= '<img class="mjpg" class="mjpg" src="/mjpg/'+name+'" alt="'+name+'" height="400"/>'
        return s
    def _web_repr_status_(self,name,namespace):
        import cgi
        s= ""
        try:
            s+= " + time: "+str(self.t)+"<br>"
        except:
            pass
        return s
def _create_mainloop():
    try:
        import Tkinter as tk
    except ImportError:
        import tkinter as tk
    global _main_root
    _main_root = tk.Tk()
    _main_root.title('Hidden Display')
    _main_root.withdraw()
    _started_tkinter_main_loop = True
    print("Starting Tk main thread...")
    _main_root.mainloop()    

def run_visualizer():
    try:
        import thread
    except ImportError:
        import _thread as thread
    import time
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

try:
    import h5py
    class HDF5Reader(Stream):
        def __init__(self,filename,data_set_name='images'):
            self.file = h5py.File(filename, "r") 
            self.data_set_name = data_set_name
            self.dset = self.file[data_set_name]
            self.i = 0
        def get(self,i=1):
            """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
            d = self.dset[self.i:self.i+i]
            self.i += i
            return d
    class HDF5Writer(Stream):
        def __init__(self,filename,data_set_name='images',size=None):
            self.filename = filename
            #self.file = h5py.File(filename, "r+") 
            self.data_set_name = data_set_name
            if size is not None:
                with h5py.File(self.filename, "a") as f:
                    self.dset = self.file.create_dataset(data_set_name, 
                                (0,size[0],size[1]), 
                                compression="gzip",
                                chunks=(100,size[0],size[1]),
                                maxshape=(None,s.shape[1],s.shape[2]),
                                dtype='f')
        def put(self,s):
            with h5py.File(self.filename, "a") as f:
                print(set(f.keys()))
                if not self.data_set_name in set(f.keys()):
                    self.dset = f.create_dataset(self.data_set_name, 
                                (0,s.shape[1],s.shape[2]),
                                compression="gzip",
                                chunks=(200,s.shape[1],s.shape[2]),
                                maxshape=(None,s.shape[1],s.shape[2]),
                                dtype='f')
                else:
                    self.dset = f[self.data_set_name]
                assert len(s.shape) == 3
                assert s.shape[1] == self.dset.shape[1]
                assert s.shape[2] == self.dset.shape[2]
                self.dset.resize(self.dset.shape[0] + s.shape[0],axis=0)
                self.dset[-s.shape[0]:,:,:] = s
except:
    class HDF5Reader(Stream):
        def __init__(self,*args,**kwargs):
            raise Exception('h5py needs to be installed!')

    class HDF5Writer(Stream):
        def __init__(self,*args,**kwargs):
            raise Exception('h5py needs to be installed!')

class NumpyReader(Stream):
    """reads a numpy file.

        .. warning::

            This class is not currently maintained and might change without notice between releases.

    """
    def __init__(self, file_ref, size=None, pixel_per_degree=10):
        self.file = file_ref
        with np.load(self.file) as data:
            if hasattr(data,'files'): 
                self.sequence = data[data.files[0]]
            else:
                self.sequence = data
            if len(self.sequence.shape) == 3:
                self.size = self.sequence.shape[1:]
            elif len(self.sequence.shape) == 5:
                self.size = self.sequence.shape[3:]
        self.i = 0
    def get(self,i=1):
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        d = self.sequence[self.i:self.i+i,:,:]
        self.i += i
        return d
    def __len__(self):
        return len(self.sequence)

class NumpyWriter(Stream):
    pass


class VideoReader(Stream):
    """
    `filename` can be either a device number or the path to a video file.
    The file will be opened as such: `cv2.VideoCapture(filename)`

    If only one camera is connected, it can be selected with `0`.
    If more than one camera is connected, 

    .. warning::

        This class is not currently maintained and might change without notice between releases.


    Parameters
    ----------
    filename (int or str):
        a string containing the url/path to a file or a number signifying a camera
    size (tuple of int, int):
        The size of the frames cut from the video
    offset (None or tuple of int, int):
        The offset of the part cut from the video
    dt (float):
        (does nothing right now)
    mode (str):
        color mode: 'mean', 'r', 'g', 'b' or 'rgb'
        Default is taking the mean over all color channels
        'rgb' will give a 5d output (does not work with all Output Streams)
        
        .. versionadded:: 0.6.4


    Examples
    --------
    
    .. plot::
        :include-source:

        import convis
        vid_in = convis.streams.VideoReader('/home/jacob/convis/input.avi',size=(200,200),mode='rgb')
        convis.plot_tensor(vid_in.get(5000)) # shows three channels


    .. code::

        import convis
        vid_in = convis.streams.VideoReader('some_video.mp4',size=(200,200),mode='mean')
        retina = convis.models.Retina()
        o = retina.run(vid_in, dt=200, max_t=2000) # run for 2000 frames

    """
    def __init__(self,filename=0,size=(50,50),offset=None,dt=1.0/24.0,mode='mean'):        
        try:
            import cv2
        except:
            print(""" OpenCV has to be installed for video input """)
            raise
        self.cv2_module = cv2
        self.size =size
        self.offset = offset
        self.mode = mode
        self.cap = cv2.VideoCapture(filename)
        self.dt = dt
        self.i = 0
        self.last_image = np.zeros(self.size)
        #frames = np.asarray([np.asarray(f) for f in v[i]])
    def __len__(self):
        return len(self.sequence)
    def get_one_frame(self):
        ret, frame = self.cap.read()
        if ret:
            offset = self.offset
            if self.offset is None:
                offset = (int(np.floor((frame.shape[0]-self.size[0])/2.0)),
                          int(np.floor((frame.shape[1]-self.size[1])/2.0)))
            if self.mode == 'mean':
                return frame[offset[0]:(offset[0]+self.size[0]),
                                    offset[1]:(offset[1]+self.size[1])].mean(2)
            if self.mode == 'r':
                return frame[offset[0]:(offset[0]+self.size[0]),
                                    offset[1]:(offset[1]+self.size[1]),0]
            if self.mode == 'g':
                return frame[offset[0]:(offset[0]+self.size[0]),
                                    offset[1]:(offset[1]+self.size[1]),1]
            if self.mode == 'b':
                return frame[offset[0]:(offset[0]+self.size[0]),
                                    offset[1]:(offset[1]+self.size[1]),2]
            if self.mode == 'rgb':
                return frame[offset[0]:(offset[0]+self.size[0]),
                                    offset[1]:(offset[1]+self.size[1])]
            raise Exception("'mode' of VideoReader not recognized! Valid choices: mean, r, g, b, rgb")
        else:
            if self.mode == 'rgb':
                return np.zeros((self.size[0],self.size[1],3))
            return np.zeros(self.size)
    def get_image(self):
        return self.last_image
    def get(self,i):
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        self.i += i
        frames = np.asarray([self.get_one_frame() for f in range(i)])
        self.last_image = frames[-1]
        if len(frames.shape) == 4:
            # moving color dimension to channels and outputting 5d
            frames = frames[None,:,:,:,:].swapaxes(0,4)[None,:,:,:,:,0]
            self.last_image = frames[0,:,-1].mean(0)
        return frames
    def close(self):
        self.cap.release()


class VideoWriter(Stream):
    """Writes data to a video

    .. warning::

        This class is not currently maintained and might change without notice between releases.


    Parameters
    ----------
    filename (str):
        Video file where output should be written to (extension determines format)
    size ((int,int) tuple):
        the size of the output
    codec (str):
        possible codecs: 'DIVX', 'XVID', 'MJPG', 'X264', 'WMV1', 'WMV2'
        (see also: `http://www.fourcc.org/codecs.php`_ )
        default: 'XVID'
    isColor (bool):
        whether the video is written in color or grey scale
        If color input is written to the file when isColor is False, the mean over all channels will be taken
        If the input is bw and isColor is True, the color channels will be duplicated (resulting in a bw image).

        .. versionadded:: 0.6.4

    Examples
    --------
    
    :class:`VideoReader` and :class:`VideoWriter` work together in color and black/white:
    
    .. code::

        import convis
        vid_in = convis.streams.VideoReader(some_video.mp4',size=(200,200),mode='rgb')
        vid_out = convis.streams.VideoWriter('test.mp4',size=(200,200),codec='X264')
        vid_out.put(vid_in.get(1000)) # copy 1000 color frames
        vid_out.close()
        vid_in.close()

    .. code::

        import convis
        vid_in = convis.streams.VideoReader('some_video.mp4',size=(200,200),mode='mean')
        vid_out = convis.streams.VideoWriter('test.mp4',size=(200,200),isColor=False)
        vid_out.put(vid_in.get(1000)) # copy 1000 black-and-white frames
        vid_out.close()
        vid_in.close()
    
    """
    def __init__(self,filename='output.avi',size=(50,50),codec='XVID', isColor=True):
        try:
            import cv2
        except:
            print(""" OpenCV has to be installed for video output """)
            raise
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = cv2.VideoWriter()
        self.out.open(filename,fourcc, fps=20.0, frameSize=size, isColor=isColor)
        self.isColor = isColor
    def put(self,s):
        if len(s.shape) == 5:
            # concatenate batches and swap channels to the last dimension
            s = np.concatenate(s,axis=1)[:,:,:,:,None].swapaxes(0,4)[0]
            if not self.isColor:
                s = s.mean(-1)
        else:
            if self.isColor:
                s = s[:,:,:,None].repeat(3,axis=-1)
        for frame in s:
            self.last_image = frame
            self.out.write(np.uint8(frame))
    def close(self):
        self.out.release()

from collections import OrderedDict as _OrderedDict

class _LimitedSizeDict(_OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        _OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        _OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)
        
class ImageSequence(Stream):
    """loads a sequence of images so they can be processed by convis

    The image values will be scaled between 0 and 256.
    
    When `is_color` is True, it returns a 5d array with dimensions [0,color_channel,time,x,y].

    If `offset` is `None`, the crop will be chosen as the center of the image.
    

    .. warning::

        This function is not stable and might change without notice between releases.
        To create stable code it is currently recommended to load input manually into numpy arrays.

    Parameters
    ----------
    filenames :  str or list
        List of filenames or a wildcard expression (eg. 'frame_*.jpeg') of the images to load.
    
    size : two tuple (int,int)
        the size the output image should be cropped to

    repeat : int
        how often each image should be repeated (eg. 100 to show a new image after 100 frames) 
    
    offset : None or (int,int)
        offset of the crop from the top left corner of the image

    is_color : bool
        Whether images should be interpreted as color or grayscale. If `is_color` is True, this stream gives a 5d output.

    Attributes
    ----------
    self.i (int) : 
        internal frame counter

    Methods
    -------

    get(i)
        retrieve a sequence of `i` frames. Advances the internal counter.


    Examples
    --------
    
    Choosing a crop with `size` and `offset`:
    
    .. code::

        import convis
        inp = convis.streams.ImageSequence('cat.png', repeat=100, size=(10,10), offset=(10,50))
    
    Creating a 500 frame input:

    .. code::

        inp = convis.streams.ImageSequence(['cat.png']*500)
        convis.plot(inp.get(100))


    .. code::

        inp = convis.streams.ImageSequence('cat.png', repeat=500)
        convis.plot(inp.get(100))


    Loading multiple images:
    
    .. code::

        inp = convis.streams.ImageSequence('frame_*.png', repeat=10)
        convis.plot(inp.get(100))

    .. code::

        inp = convis.streams.ImageSequence(['cat.png','dog.png','parrot.png'], repeat=200)
        convis.plot(inp.get(100))
        
        

    """
    def __init__(self,filenames='*.jpg',size=(50,50),repeat=1,offset=None,is_color=False):
        self.repeat = repeat
        self.offset = offset
        self.cache = _LimitedSizeDict(size_limit=20)
        self.is_color = is_color
        self.i=0
        if type(filenames) is str:
            import glob
            self.file_list = sorted(glob.glob(filenames))
        elif type(filenames) is list:
            self.file_list = filenames
        else:
            raise Exception('filenames not recognized. Has to be a string or list.')
        super(ImageSequence,self).__init__()
        self.size = size
    def __str__(self):
        return "convis.streams.ImageSequence size: "+str(self.size)+" [frame "+str(self.i)+"/"+str(len(self))+"]"
    def __repr__(self):
        return "<convis.streams.ImageSequence size="+str(self.size)+", [frame "+str(self.i)+"/"+str(len(self))+"]>"
    def _repr_html_(self):
        from . import variable_describe
        def _plot(fn):
            from PIL import Image
            try:
                import matplotlib.pylab as plt
                t = np.array(Image.open(fn))
                plt.figure()
                plt.imshow(self._crop(t))
                plt.axis('off')
                return "<img src='data:image/png;base64," + variable_describe._plot_to_string() + "'>"
            except:
                return "<br/>Failed to open."
        s = "<b>ImageSequence</b> size="+str(self.size)
        s += ", offset = "+str(self.offset)
        s += ", repeat = "+str(self.repeat)
        s += ", is_color = "+str(self.is_color)
        s += ", [frame "+str(self.i)+"/"+str(len(self))+"]"
        s += "<div style='background:#ff;padding:10px'><b>Input Images:</b>"
        for t in np.unique(self.file_list)[:10]:
            s += "<div style='background:#fff; margin:10px;padding:10px; border-left: 4px solid #eee;'>"+str(t)+": "+_plot(t)+"</div>"
        s += "</div>"
        return s
    def __setattr__(self,k,v):
        if k in ['is_color']:
            # 'size' and 'offset' don't affect the cache
            self.cache = _LimitedSizeDict(size_limit=20)
        super(ImageSequence,self).__setattr__(k,v)
    def __len__(self):
        file_list = list(np.repeat(self.file_list,self.repeat))
        return len(file_list)
    def _crop(self,frame):
        offset = self.offset
        if self.offset is None:
            offset = (int(np.floor((frame.shape[0]-self.size[0])/2.0)),
                      int(np.floor((frame.shape[1]-self.size[1])/2.0)))
        return frame[offset[0]:(offset[0]+self.size[0]),
                     offset[1]:(offset[1]+self.size[1])]  
    def get_one_frame(self):
        file_list = list(np.repeat(self.file_list,self.repeat))
        if file_list[self.i] in self.cache.keys():
            frame = self.cache[file_list[self.i]]
        else:
            from PIL import Image
            frame = np.array(Image.open(file_list[self.i]))
            if not self.is_color and len(frame.shape) > 2:
                frame = frame.mean(2)
            if self.is_color:
                if len(frame.shape) == 2:
                    print('2 to 3d')
                    frame = frame[:,:,None]
                if frame.shape[2] < 3:
                    frame = np.repeat(frame[:,:,None],3,axis=2)
                if frame.shape[2] > 3:
                    frame = frame[:,:,:3]
            self.cache[file_list[self.i]] = frame
        cropped_frame = self._crop(frame)
        self.last_image = cropped_frame
        self.i += 1
        return cropped_frame
    def get_image(self):
        return self.last_image
    def get(self,i):
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        if self.i + i > len(self):
            raise Exception('ImageSequence is at its end.'+str(self.i)+'+'+str(i)+' >= '+str(len(self)))
        frames = np.asarray([self.get_one_frame() for f in range(i)])
        if self.is_color:
            return np.moveaxis(frames,3,0)[None,:,:,:,:]
        else:
            return frames


class MNISTStream(Stream):
    """Downloads MNIST data and displays random digits.
    
        The digits are chosen pseudo randomly using a starting value `init` and a (large) advancing value.
        If both values are supplied, the stream will return the same sequence of numbers.
        By default, `init` is chosen randomly and `advance` is set to an arbitrary (but fixed) high prime number.
        If `advance` shares factors with the number of samples in MNIST, not all examples will be covered!
    
        If `include_label` is True, the output will contain a second channel with the matching label.
        (Currently, there is no automated way to pass this along with the other channel)

        .. note::

            The way the label is encoded might change depending on how people want to use it.
            Please `submitting a github issue if you have in idea how to pass labels along with the input <https://github.com/jahuth/convis/labels/enhancement>`_!

        The output of the stream will be frames of `28 x 28` pixels.

        .. versionadded:: 0.6.4


        Parameters
        ----------
        data_folder : str
            A (relative or absolute) path where the MNIST data should be downloaded to.
            If a valid download is found at that location, the existing data will be used.
            This stream uses `torchvision` to download and extract the data, more info about the downloader can be found `here <https://pytorch.org/docs/stable/torchvision/datasets.html#mnist>`_.
        rep : int
            Number of frames each digit should be repeated
        max_value : float
            The stimuli will have normalized values between 0 and `max_value`.
        init : int or None
            The starting point of the pseudo-random walk through the examples.
            If the value is `None`, it is chosen randomly with uniform probability.
        advance : int
            A large number that determines the next sample.
            If this value is not touched, it will generate pseudo-random walks starting deterministically from `init`.
        include_label : bool
            Whether to include an additional channel carrying the labels or not (default: `False`)
        buffer : `torch.Tensor`
            the MNIST dataset

        Attributes
        ----------
        i : int
            index of current image.
        j : int
            number of repetitions already displayed (secondary index)
        init_i  : int
            The initial starting point of the walk.


        Examples
        --------

        .. plot::
            :include-source:

            import convis
            from matplotlib.pylab import plot, xlim, gcf
            stream = convis.streams.MNISTStream('../data',rep=20)
            convis.plot(stream.get(500))
            gcf().show()
            convis.plot(stream.get(500))
            gcf().show()
            stream.reset() # returning to the start
            convis.plot(stream.get(500))
            gcf().show()

        See Also
        --------

        convis.streams.PoissonMNISTStream
        convis.streams.PseudoNMNIST

    """
    def __init__(self,data_folder='./mnist_data',rep = 10,max_value=1.0,
                 init=None,advance=179425529,include_label=False):
        import torch
        try:
            from torchvision import datasets, transforms
        except:
            raise Exception("The package 'torchvision' needs to be installed to use the MNIST dataset.")
        if init is None:
            from numpy.random import randint
            init = randint(10**10)
        self.include_label = include_label
        self.init_i = init
        self.max_value = max_value
        self.rep = rep
        self.advance = advance
        training_data = datasets.MNIST(data_folder, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.buffer = torch.load(data_folder+'/processed/test.pt')
        self.reset()
    def reset(self):
        """Return the random walk to the starting point (`self.init_i`)."""
        self.i = self.init_i%len(self.buffer[1])
        self.j = 0
    def get(self,i=1):
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        a = []
        b = []
        if self.j > 0:
            a.append(self.max_value*(self.buffer[0][self.i].float()/255.0)[None,None,None,:,:].repeat(1,1,self.j,1,1))
            b.append(self.buffer[1][self.i].float().repeat(self.j,28,28)[None,None,:])
        while i > self.j:
            self.j += self.rep
            self.i = (self.i+self.advance)%len(self.buffer[1])
            a.append(self.max_value*(self.buffer[0][self.i].float()/255.0)[None,None,None,:,:].repeat(1,1,self.rep,1,1))
            b.append(self.buffer[1][self.i].float().repeat(self.rep,28,28)[None,None,:])
        a = np.concatenate(a,axis=2)
        b = np.concatenate(b,axis=2)
        self.j = (self.j-i)%self.rep
        if self.include_label:
            return np.concatenate([a,b],axis=0)[:,:,:i,:,:]
        return a[:,:,:i,:,:]


class PoissonMNISTStream(Stream):
    """Downloads MNIST data and generates Poisson spiking for random digits.
    
        This definition of a neuromorphic version of the MNIST data set follows eg.:

          - P. Diehl and M. Cook. Unsupervised learning of digit recognition using spike-timing-dependent plasticity. Frontiers in Computational Neuroscience, 9(99), 2015.
          - D. Querlioz, O. Bichler, P. Dollfus, and C. Gamrat. Immunity to device variations in a spiking neural network with memristive nanodevices. IEEE Transactions on Nanotechnology, 12:288-295, 2013.

        An alternative interpretation are events that are created by saccade-like movements over the images.

        The digits are chosen pseudo randomly using a starting value `init` and a (large) advancing value.
        If both values are supplied, the stream will return the same sequence of numbers.
        By default, `init` is chosen randomly and `advance` is set to an arbitrary (but fixed) high prime number.
        If `advance` shares factors with the number of samples in MNIST, not all examples will be covered!
    
        If `include_label` is True, the output will contain a second channel with the matching label.
        (Currently, there is no automated way to pass this along with the other channel)


        The output of the stream will be frames of `28 x 28` pixels of binary coded spike trains.

        .. versionadded:: 0.6.4


        Parameters
        ----------
        data_folder : str
            A (relative or absolute) path where the MNIST data should be downloaded to.
            If a valid download is found at that location, the existing data will be used.
            This stream uses `torchvision` to download and extract the data, more info about the downloader can be found `here <https://pytorch.org/docs/stable/torchvision/datasets.html#mnist>`_.
        rep : int
            Number of frames each digit should be repeated
        fr : float
            The firing probability at the highest possible value.
        init : int or None
            The starting point of the pseudo-random walk through the examples.
            If the value is `None`, it is chosen randomly with uniform probability.
        advance : int
            A large number that determines the next sample.
            If this value is not touched, it will generate pseudo-random walks starting deterministically from `init`.
        include_label : bool
            Whether to include an additional channel carrying the labels or not (default: `False`)
        buffer : `torch.Tensor`
            the MNIST dataset

        Attributes
        ----------
        i : int
            index of current image.
        j : int
            number of repetitions already displayed (secondary index)
        init_i  : int
            The initial starting point of the walk.

        Examples
        --------

        .. plot::
            :include-source:

            import convis
            from matplotlib.pylab import plot, xlim, gcf
            stream = convis.streams.PoissonMNISTStream('../data',rep=20,fr=0.20) 
            # here we are using a very high firing rate for easy visualization
            # (20% of cells are active in each frame)
            convis.plot(stream.get(500))
            gcf().show()
            convis.plot(stream.get(500))
            gcf().show()
            stream.reset() # returning to the start
            convis.plot(stream.get(500))
            gcf().show()

        See Also
        --------

        convis.streams.MNISTStream
        convis.streams.PseudoNMNIST

    """
    def __init__(self,data_folder='./mnist_data',rep = 10,fr=0.05,
                 init=None,advance=179425529,include_label=False):
        import torch
        from .filters.spiking import Poisson
        try:
            from torchvision import datasets, transforms
        except:
            raise Exception("The package 'torchvision' needs to be installed to use the MNIST dataset.")
        if init is None:
            from numpy.random import randint
            init = randint(10**10)
        self.include_label = include_label
        self.init_i = init
        self.fr = fr
        self.rep = rep
        self.advance = advance
        training_data = datasets.MNIST(data_folder, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        self.buffer = torch.load(data_folder+'/processed/test.pt')
        self.reset()
        self.poisson = Poisson()
    def reset(self):
        """Return the random walk to the starting point (`self.init_i`)."""
        self.i = self.init_i%len(self.buffer[1])
        self.j = 0
    def get(self,i=1):
        """outputs the next slice of length `i` in the stream (see :meth:`convis.streams.Stream`)"""
        a = []
        b = []
        if self.j > 0:
            a.append(self.poisson(self.fr*(self.buffer[0][self.i].float()/255.0)[None,:,:].repeat(self.j,1,1)))
            b.append(self.buffer[1][self.i].float().repeat(self.j,28,28)[None,None,:])
        while i > self.j:
            self.j += self.rep
            self.i = (self.i+self.advance)%len(self.buffer[1])
            a.append(self.poisson(self.fr*(self.buffer[0][self.i].float()/255.0)[None,:,:].repeat(self.rep,1,1)))
            b.append(self.buffer[1][self.i].float().repeat(self.rep,28,28)[None,None,:])
        a = np.concatenate(a,axis=2)
        b = np.concatenate(b,axis=2)
        self.j = (self.j-i)%self.rep
        if self.include_label:
            return np.concatenate([a,b],axis=0)[:,:,:i,:,:]
        return a[:,:,:i,:,:]

class PseudoNMNIST(ProcessingStream):
    """A motion driven conversion of MNIST into spikes. A quick way to get spiking input.
    
    Internally, it defines a Layer that performs three predetermined "saccades" (see the source code if you want to use it as a Saccade generator in other contexts).
    It uses a :class:`~convis.filters.spiking.IntegrativeMotionSensor` to generate spikes from the movement.

    .. versionadded:: 0.6.4

    Parameters
    ----------
    mnist_data_folder : str
        Where to find (or download to) the MNIST dataset. (see :class:`convis.streams.MNISTStream`)
    threshold : float
        the sensitivity of the motion detection (smaller values are more sensitive)
    output_size : tuple(int,int)
        size of the generated output (the data is padded with zeros)

    Examples
    ---------


    .. plot::
        :include-source:

        import convis
        n = convis.streams.PseudoNMNIST()
        convis.plot(n.get(200)) 

    """
    def __init__(self,mnist_data_folder='../data/',threshold=5.0,output_size=(34,34)):
        from . import base, filters
        class ThreeSaccadeGenerator(base.Layer):
            """A private Layer class that performs three predefined saccades"""
            def __init__(self,output_size=(28,28),padding_mode='constant'):
                self.dim = 5
                super(ThreeSaccadeGenerator,self).__init__()
                self.padding_mode = padding_mode
                self.output_size = output_size
                self.next_saccade_in = 0
                self.saccade_frequency_mean = 100.0
                self.saccade_frequency_sd = 0.0
                self.saccade_distance_mean = 5.0
                self.saccade_distance_sd = 0.0
                self.register_state('pos',np.array([28,28]))
                self.t = 0
                self.path = np.concatenate([[np.linspace(0.0,1.0,100),np.linspace(0.0,1.0,100)],
                             [np.linspace(1.0,0.0,100),np.linspace(1.0,0.0,100)],
                             [np.linspace(0.0,0.0,100),np.linspace(1.0,0.0,100)]],axis=1)
            def forward(self,inp):
                import torch
                max_x, max_y = min(inp.size()[3],self.output_size[0]),min(inp.size()[4],self.output_size[1])
                padding = (max_x,2*max_x,max_y,2*max_y,0,0)
                inp_padded = torch.nn.functional.pad(inp,padding,self.padding_mode, 0)
                inps = []
                for t in range(inp.size()[2]):
                    self.pos = 0.75*inp.size()[3]*self.path[:,self.t%(self.path.shape[1])]
                    self.t += 1
                    x = int((int(self.pos[0])%(inp.size()[3]+max_x))+0.75*max_x)
                    y = int((int(self.pos[1])%(inp.size()[4]+max_y))+0.75*max_y)
                    inps.append(inp_padded[:,:,t:(t+1),x:(x+self.output_size[0]),y:(y+self.output_size[0])])
                return torch.cat(inps,dim=2)
            
        super(PseudoNMNIST,self).__init__(MNISTStream(mnist_data_folder,rep=300), 
                                    model = filters.spiking.IntegrativeMotionSensor('linear',threshold), 
                                    pre = ThreeSaccadeGenerator(output_size=output_size))

