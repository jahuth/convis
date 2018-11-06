import convis
vid_in = convis.streams.VideoReader('/home/jacob/convis/input.avi',size=(200,200),mode='rgb')
convis.plot_tensor(vid_in.get(5000)) # shows three channels