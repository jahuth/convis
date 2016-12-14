
MAX_LOG_ENTRIES = 10000

"""

Testing the runner::

    %pylab inline
    import litus
    import convis.runner
    import convis.streams
    A = np.array([1.0*(np.arange(50*50) > i).reshape((50,50)) for i in range(50*50 + 1)])
    inp = convis.streams.RepeatingStream(A,size=(50,50))
    out = convis.streams.run_visualizer()
    out.refresh_rate = 5
    f = lambda x: [np.sqrt(x)]
    r = convis.runner.Runner(f,inp,out,dt=0.005)


Visualize the computation vs real time::

    # r is the runner object
    import matplotlib.dates as mdates
    tl = litus.ld2dl(r.time_log)
    plot(np.array(tl['now']),np.array(tl['total_loop']),'ro')
    plot([np.min(tl['now']),np.max(tl['now'])],[r.dt,r.dt],'r--',lw=2)
    gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    gcf().autofmt_xdate()
    figure()
    plot(np.array(tl['now']),np.array(tl['t_real'])-np.array(tl['t']),'o')
    gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    gcf().autofmt_xdate()

"""

_debug_runner_list = []
def _debug_list_runners():
    for i,d in enumerate(_debug_runner_list):
        r = d()
        if r is not None:
            print i,'stopped' if r.closed else 'running'
def stopall():
    for i,d in enumerate(_debug_runner_list):
        r = d()
        if r is not None:
            r.stop()
            print i,'stopped' if r.closed else 'running'
class Runner(object):
    def __init__(self,func,input_stream,output_stream,**kwargs):
        self.func = func
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.t = 0.0
        self.t_zero = 0.0
        self.dt = 0.001
        self.max_t = None
        self.chunk_shape = (100,None,None)
        self.waiting_for_input_delay = 0.1
        self.realtime = True
        self.strict_realtime = True
        self.max_realtime_deviation = 10.0 # seconds
        self.max_realtime_advance = 0.5 # we can be half a second early
        self.time_log = []
        self.log = True
        self.__dict__.update(kwargs)
        self.closed = True
        global _debug_runner_list
        import weakref
        _debug_runner_list.append(weakref.ref(self))
        self.start()
    @property
    def chunk_length(self):
        return self.dt * self.chunk_shape[0]
    def deviation(self):
        import datetime
        return (datetime.datetime.now() - self.start_time).total_seconds() - (self.t-self.t_zero)
    def stop(self):
        self.closed = True
    def start(self):
        if self.closed:
            self.t_zero = self.t
            import thread
            import time, datetime
            self.closed = False
            self.start_time = datetime.datetime.now()
            thread.start_new_thread(self.thread,tuple())
    def thread(self):
        import time, datetime
        while not self.closed:
            if self.realtime:
                t_real = (datetime.datetime.now() - self.start_time).total_seconds()
                if t_real < (self.t-self.t_zero) - self.max_realtime_advance:
                    #print 'too fast, waiting for time to catch up',t_real, (self.t-self.t_zero)
                    time.sleep((self.t-self.t_zero)-(t_real- self.max_realtime_advance))
                if t_real > (self.t-self.t_zero) + self.max_realtime_deviation:
                    #print 'too slow! Have to skip some computations!',t_real, (self.t-self.t_zero)
                    if self.strict_realtime:
                        self.closed = True
                        raise Exception('Realtime could not be met!')
            if self.closed:
                break
            if self.max_t is None or (self.t-self.t_zero) + self.dt*self.chunk_shape[0] <= self.max_t and self.input_stream.available(self.chunk_shape[0]):
                t_begin = (self.t-self.t_zero)
                t_end = (self.t-self.t_zero) + self.dt*self.chunk_shape[0]
                t_real_begin = (datetime.datetime.now() - self.start_time).total_seconds()
                self.last_get_input_time = datetime.datetime.now()
                inp = self.input_stream.get(self.chunk_shape[0])
                self.last_start_computation_time = datetime.datetime.now()
                out = self.func(inp) # todo: spatial chunking
                self.last_stop_computation_time = datetime.datetime.now()
                # we assume the function always gives a list of outputs
                self.t = self.t_zero + t_end
                if type(self.output_stream) == list:
                    for (os,ot) in zip(self.output_stream,out):
                        os.put(ot)
                else:
                    self.output_stream.put(out[0])
                self.last_put_output_time = datetime.datetime.now()
                t_real_end = (datetime.datetime.now() - self.start_time).total_seconds()
                if self.log:
                    self.time_log.append({'total_loop': (self.last_put_output_time - self.last_get_input_time).total_seconds(),
                                          'get_input':  (self.last_start_computation_time - self.last_get_input_time).total_seconds(),
                                          'compute':    (self.last_stop_computation_time - self.last_start_computation_time).total_seconds(),
                                          'put_output': (self.last_put_output_time - self.last_stop_computation_time).total_seconds(),
                                          't':          [t_begin, t_end],
                                          't_real':     [t_real_begin, t_real_end],
                                          'now':         datetime.datetime.now()})
                if len(self.time_log) > 2*MAX_LOG_ENTRIES:
                    self.time_log = self.time_log[-MAX_LOG_ENTRIES:]
                #print "Processed input!"
                time.sleep(0.5)
            else:
                #print "Waiting for input..."
                time.sleep(self.waiting_for_input_delay)
