from __future__ import print_function
import socket
import sys
import StringIO
import PIL
from PIL import Image, PngImagePlugin
import cStringIO
import socket
import sys, os
import thread
import numpy as np
from future.utils import iteritems as _iteritems

def pngsave(A, file, info={}):
    """
        wrapper around PIL 1.1.7 Image.save to preserve PNG metadata
        based on public domain script by Nick Galbreath                                                                                                        

        http://blog.modp.com/2007/08/python-pil-and-png-metadata-take-2.html                                                                 
    """
    from PIL import Image, PngImagePlugin
    im = Image.fromarray(256.0*A).convert('RGB')
    reserved = ('interlace', 'gamma', 'dpi', 'transparency', 'aspect')
    meta = PngImagePlugin.PngInfo()
    for k,v in _iteritems(info):
        if k in reserved: continue
        meta.add_text(k, v, 0)
    im.save(file, "PNG", pnginfo=meta)

def pngload(file):
    """
        Returns a read png file.

        Meta information is available as `.info` attribute.

        Can be turned into a numpy array by casting (depends on PIL version)::

            numpy.array(im)

    """     
    from PIL import Image
    im = Image.open(file)
    return im


def png_server():
    import socket
    import sys
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 10000)
    print('starting up on %s port %s' % server_address, file=sys.stderr)
    sock.bind(server_address)
    sock.listen(1)

    while True:
        # Wait for a connection
        print('waiting for a connection', file=sys.stderr)
        connection, client_address = sock.accept()
        try:
            print('connection from', client_address, file=sys.stderr)

            # Receive the data in small chunks and retransmit it
            while True:
                data = connection.recv(16)
                print('received "%s"' % data, file=sys.stderr)
                if data:
                    print('sending data back to the client', file=sys.stderr)
                    connection.sendall(data)
                else:
                    print('no more data from', client_address, file=sys.stderr)
                    break
                
        finally:
            # Clean up the connection
            connection.close()

def png_client(images,info={},port=10000,host='localhost',compress_level=0,resize=(1.0,1.0)):
    if len(images.shape) == 2:
        images = [images]
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = (host, port)
    sock.connect(server_address)
    try:
        for A in images:
            im = Image.fromarray(256.0*A).convert('RGB')
            if resize != (1.0,1.0) and resize is not None:
                if not type(resize) == tuple:
                    resize = (resize,resize)
                im = im.resize((int(resize[0]*im.size[0]), int(resize[1]*im.size[1])), PIL.Image.ANTIALIAS)
            output = StringIO.StringIO()
            meta = PngImagePlugin.PngInfo()
            reserved = ('interlace', 'gamma', 'dpi', 'transparency', 'aspect')
            for k,v in _iteritems(info):
                if k in reserved:
                    continue
                meta.add_text(str(k), str(v), 0)
            im.save(output, format="PNG", pnginfo=meta, compress_level=compress_level)
            message = output.getvalue()
            output.close()
            sock.sendall(message)
    finally:
        sock.close()


def test_png_file_client(filename):
    import socket
    import sys

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('localhost', 10000)
    print('connecting to %s port %s' % server_address, file=sys.stderr)
    sock.connect(server_address)
    try:
        
        # Send data
        with open(filename,'rb') as f:
            message = f.read()
        print('sending "%s"' % filename, file=sys.stderr)
        sock.sendall(message)
    finally:
        print('closing socket', file=sys.stderr)
        sock.close()

def test_text_client(text):
    import socket
    import sys

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('localhost', 10000)
    print('connecting to %s port %s' % server_address, file=sys.stderr)
    sock.connect(server_address)
    try:
        
        # Send data
        message = text
        print('sending "%s"' % message, file=sys.stderr)
        sock.sendall(message)

        # Look for the response
        amount_received = 0
        amount_expected = len(message)
        
        while amount_received < amount_expected:
            data = sock.recv(16)
            amount_received += len(data)
            print('received "%s"' % data, file=sys.stderr)

    finally:
        print('closing socket', file=sys.stderr)
        sock.close()

class ImageRecieverServer():
    def __init__(self, port = 10000, size=(0,0)):
        from PIL import Image, ImageTk
        self.port = port
        self.size = size
        self.buffer = cStringIO.StringIO()
        self.image = Image.fromarray(np.zeros((200,200))).convert('RGB')
        self.closed = False
        self.dirty = False
        self.refresh_rate = 10
        self.last_buffer = []
        self.debug = False
        thread.start_new_thread(self.wait_for_connection,tuple())
    def __del__(self):
        self.closed = True
        self.sock.close()
        del self.sock
    def wait_for_connection(self):
        import socket
        import sys
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_address = ('localhost', self.port)
        #print('starting up on %s port %s' % self.server_address, file=sys.stderr)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #self.sock.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)
        self.sock.bind(self.server_address)
        self.sock.listen(10)
        #self.sock.setblocking(0)
        while self.closed is False:
            try:
                connection, client_address = self.sock.accept()
                if self.debug:
                    print("Accepted connection.")
                thread.start_new_thread(self.serve,(connection, client_address))
            except Exception as e:
                print(e)
                raise
        self.sock.close()
        self.closed = True
    def serve(self,connection, client_address):
        try:
            if self.debug:
                print("Recieving Data.")
            dirty = False
            all_data = cStringIO.StringIO()
            last_data = b""
            while True:
                data = connection.recv(16)
                dirty = True
                all_data.write(data)
                if b"IEND" in last_data+data:
                    if self.debug:
                        print("Recieved End of Image.")
                    #print (last_data+data)
                    #print 'found IEND!'
                    self.last_buffer.append(all_data)
                    all_data = cStringIO.StringIO()
                    all_data.write((last_data+data)[(last_data+data).find(b"IEND")+8:])
                    dirty = False
                    last_data = (last_data+data)[(last_data+data).find(b"IEND")+8:]
                else:
                    last_data = data
                #self.buffer.write(data)
                #all_data += data
                if not data:
                    break
            if dirty:
                #print 'Cleaning up unclosed image...'
                self.last_buffer.append(all_data)
            #self.dirty = True # redraw the image
        finally:
            if self.debug:
                print("Closing connection.")
            connection.close()            
    def available(self):
        return len(self.last_buffer) > 0
    def __len__(self):
        return len(self.last_buffer)
    def get(self,n=1):
        if n > len(self.last_buffer):
            raise Exception('Not ready yet.')
        images = []
        for i in range(n):
            images.append(self.get_image())
        return np.array(images)
    def get_image(self):
        from PIL import Image, ImageTk
        refresh = self.refresh_rate
        if len(self.last_buffer) > 0:
            try:
                image_buffer = self.last_buffer.pop()
                self.image = Image.open(image_buffer)
                self.image.load()
                return np.array(self.image).mean(2)
            except Exception as e:
                return np.zeros(self.size)
                #if self.available():
                #    # maybe only this image was bad?
                #    return self.get_image()
                #pass


def png_display():
    import Tkinter as tk
    from PIL import Image, ImageTk
    from ttk import Frame, Button, Style
    import time

    class Example():
        def __init__(self):
            self.root = tk.Tk()
            self.root.title('Display')
            self.buffer = cStringIO.StringIO()
            self.image = Image.fromarray(np.zeros((200,200))).convert('RGB')
            self.image1 = ImageTk.PhotoImage(self.image)
            self.dirty = False
            self.refresh_rate = 10
            self.last_buffer = []
            self.panel1 = tk.Label(self.root, image=self.image1)
            self.display = self.image1
            self.panel1.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
            self.root.after(100, self.update_image)
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.closed = False
            self.recieve_thread = thread.start_new_thread(self.recieve,tuple())
            self.root.mainloop()
        def on_closing(self):
            self.closed = True
            self.root.destroy()
            self.sock.close()
            del self.sock
        def recieve(self):
            import socket
            import sys
            # Create a TCP/IP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            port = 10000
            for i in range(200):
                try:
                    self.server_address = ('localhost', port)
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    #self.sock.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)
                    self.sock.bind(self.server_address)
                except:
                    #print 'port taken. starting up on %s port %s' % self.server_address
                    port = 10000 + np.random.randint(1000)
            self.root.title('%s:%s' % self.server_address)
            print('starting up on %s port %s' % self.server_address, file=sys.stderr)
            #self.sock.setblocking(0)
            self.sock.listen(10)
            while self.closed is False:
                try:
                    print('waiting...')
                    connection, client_address = self.sock.accept()
                    thread.start_new_thread(self.serve,(connection, client_address))
                except:
                    raise
                    pass
            self.sock.close()
        def serve(self,connection, client_address):
            try:
                dirty = False
                all_data = cStringIO.StringIO()
                last_data = b""
                while True:
                    data = connection.recv(16)
                    dirty = True
                    all_data.write(data)
                    if b"IEND" in last_data+data:
                        #print (last_data+data)
                        #print 'found IEND!'
                        self.last_buffer.append(all_data)
                        all_data = cStringIO.StringIO()
                        all_data.write((last_data+data)[(last_data+data).find(b"IEND")+8:])
                        dirty = False
                        last_data = (last_data+data)[(last_data+data).find(b"IEND")+8:]
                    else:
                        last_data = data
                    #self.buffer.write(data)
                    #all_data += data
                    if not data:
                        break
                if dirty:
                    print('Cleaning up unclosed image...')
                    self.last_buffer.append(all_data)
                #self.dirty = True # redraw the image
            finally:
                connection.close()            
        def update_image(self):
            refresh = self.refresh_rate
            if len(self.last_buffer) > 0:
                try:
                    #print len(self.last_buffer)
                    image_buffer = self.last_buffer.pop()
                    #print len(self.buffer.read())
                    #self.buffer.seek(0)
                    self.image = Image.open(image_buffer)#cStringIO.StringIO(self.last_buffer))
                    self.image.load()
                    #print len(self.buffer.read())
                    #new_buffer = cStringIO.StringIO()
                    #new_buffer.write(self.buffer.read())
                    #self.buffer = new_buffer
                    #self.dirty = False
                    self.image1 = ImageTk.PhotoImage(self.image)
                    self.panel1.configure(image=self.image1)
                    self.root.title(str(len(self.last_buffer))+' Images buffered')
                    self.display = self.image1
                    refresh = self.image.info.get('refresh',refresh)
                except Exception as e:
                    print(e)
                    pass
            self.root.after(refresh, self.update_image)

    app = Example()