
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
    def __init__(self,func,input_stream,output_stream,realtime=False,strict_realtime=False,**kwargs):
        self.func = func
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.t = 0.0
        self.t_zero = 0.0
        self.dt = 0.001
        self.refresh_rate = 0.01
        self.max_t = None
        self.chunk_shape = (100,None,None)
        self.waiting_for_input_delay = 0.1
        self.realtime = realtime
        self.strict_realtime = strict_realtime
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
        if self.func is None or self.input_stream is None or self.output_stream is None:
            return False
        if self.closed:
            self.t_zero = self.t
            import thread
            import time, datetime
            self.closed = False
            self.start_time = datetime.datetime.now()
            thread.start_new_thread(self.thread,tuple())
            return True
    def _web_set_(self,slot,value, namespace={}):
        if value in namespace.keys():
            if slot == 'input':
                if hasattr(namespace[value],'get'):
                    self.input_stream = namespace[value]
                    return 'ok.'
            if slot == 'output':
                if hasattr(namespace[value],'put'):
                    self.output_stream = namespace[value]
                    return 'ok.'
            if slot == 'func':
                if hasattr(namespace[value],'run'):
                    self.func = namespace[value].run
                    return 'ok.'
                elif hasattr(namespace[value], '__call__'):
                    self.func = namespace[value]
                    return 'ok.'
    def _web_action_(self,action, namespace={}):
        if action == 'start':
            if self.start():
                return 'ok.'
            return 'Runner is not fully configured.'
        if action == 'stop':
            self.stop()
            return 'ok.'
    def _web_repr_(self, name, namespace={}):
        def get_name(t):
            if t in [None] or type(t) in [int,float,str]:
                return str(t)
            import cgi
            for k,v in namespace.items():
                try:
                    if t is v:
                        return k
                except:
                    pass
            try:
                return cgi.escape(t.__name__)
            except:
                return cgi.escape(str(t))
        s = """This is a runner object<br>
        Function: <select class='namespace_select' data-slot='func'><option>"""+get_name(self.func)+"""</option></select><br>
        Input: <select class='namespace_select' data-slot='input'><option>"""+get_name(self.input_stream)+"""</option></select><br>
        Output: <select class='namespace_select' data-slot='output'><option>"""+get_name(self.output_stream)+"""</option></select><br>
        <button onclick='action("start");'>start</button> <button onclick='action("stop");'>stop</button>"""
        return s
    def _web_repr_status_(self, name, namespace={}):
        return """
            Status: """+str("Stopped." if self.closed else "Running.")+"""<br>
            Time: """+str(self.t)+""" Seconds elapsed<br>
            """
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
                #print self.t
                self.last_get_input_time = datetime.datetime.now()
                inp = self.input_stream.get(self.chunk_shape[0])
                self.last_start_computation_time = datetime.datetime.now()
                out = self.func(inp) # todo: spatial chunking
                self.last_stop_computation_time = datetime.datetime.now()
                # we assume the function always gives a list of outputs
                self.t = self.t_zero + t_end
                if type(self.output_stream) == list:
                    for (outs,ot) in zip(self.output_stream,out):
                        outs.put(ot)
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
                time.sleep(max(self.refresh_rate-t_real_end,0.0))
                if type(self.output_stream) == list:
                    for outs in self.output_stream:
                        if hasattr(outs,'buffered'):
                            while outs.buffered > self.chunk_shape[0]*3:
                                time.sleep(self.waiting_for_input_delay)
                else:
                    if hasattr(self.output_stream,'buffered'):
                        while self.output_stream.buffered > self.chunk_shape[0]*3:
                            time.sleep(self.waiting_for_input_delay)
            else:
                #print "Waiting for input..."
                time.sleep(self.waiting_for_input_delay)
