from __future__ import print_function
import convis
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pylab as plt

import convis.streams

namespace = {'convis': convis, 'np': np, 'plt':plt}
namespace['input'] = convis.streams.RepeatingStream(np.random.rand(100,50,50),size=(50,50))
namespace['runner'] = convis.base.Runner(None,None,None)
namespace['tmp_output'] = convis.streams.SequenceStream()

def process_argument(arg,i,argv):
    global namespace
    if arg in ['help','--help','-h']:
        print("""
Convis command line tool

    python -m convis.__main__ script.py run

Execute script and subsequently start a runner.

    python -m convis.__main__ config.json run

Create a model from a json file and subsequently start a runner.

    python -m convis.__main__ file.inr animate

Load an inr file and display it as an animation (slowed down by a factor of x10)

""")
    if arg.endswith('.py') and not arg.endswith('__main__.py'):
        l = {}
        execfile(arg, {}, namespace)
    if arg.endswith('.json'):
        with open(sys.argv[1],'r') as f:
            namespace.update(namespace['convis'].load_json(f.read()))
    if arg == 'random':
        namespace['input'] = namespace['convis'].streams.RandomStream(size=(50,50))
    if arg.endswith('.inr'):
        namespace['input'] = namespace['convis'].streams.InrImageStreamer(sys.argv[1],z=False)
    if arg == 'animate':
        v = namespace['convis'].streams.run_visualizer()
        v.auto_scale_refresh_rate = True
        v.decay = 0.2
        v.refresh_rate = 0.01
        v.minimal_refresh_rate = 0.01
        namespace['v'] = v
        import time
        while True:
            try:
                v.put(namespace['input'].get(100)[:,:,:])
            except:
                break
            time.sleep(1.0)
    if arg == 'runner':
        if namespace.get('model',None) is None:
            raise Exception('No model loaded!')
        if namespace.get('output',None) is None:
            namespace['output'] = namespace['convis'].streams.run_visualizer()
        namespace['runner'] = namespace['convis'].base.Runner(namespace.get('model').run,namespace.get('input'),namespace.get('output'),dt=0.001)
        namespace['start'] = namespace['runner'].start
        namespace['stop'] = namespace['runner'].stop
    if arg == 'stop': 
        if 'runner' in namespace.keys():
            namespace['runner'].stop()     
    if arg == 'start': 
        if 'runner' in namespace.keys():
            namespace['runner'].start()     
    if arg == 'server':      
        while True:
            key_press = raw_input(" > ")
            if key_press[0] == 'a':
                namespace['runner'].start()
            if key_press[0] == 's':
                namespace['runner'].stop()
            if key_press[0] == 'q':
                break
    if arg == 'output_inr':
        namespace['output'] = namespace['convis'].streams.InrImageStreamWriter('output.inr',v=1,x=50,y=50,z=1)
        with namespace['output']:
            r = namespace['convis'].base.Runner(namespace['model'].run,namespace['input'],namespace['output'],dt=0.001)
            while True:
                key_press = raw_input(" > ")
                if key_press[0] == 'a':
                    r.start()
                if key_press[0] == 's':
                    r.stop()
                if key_press[0] == 'q':
                    break
            r.stop()
            print('Updating header of inr file...')
            time.sleep(1.0)
    if arg == 'console' or arg == 'ipython':
        try:
            import IPython
            from IPython.config.loader import Config
            cfg = Config()
            cfg.TerminalInteractiveShell.banner1 = '      Convis ipython shell'
            cfg.TerminalInteractiveShell.banner2 = '''            (>--
    Use `\x1b[31mstart()\x1b[0m` and `\x1b[31mstop()\x1b[0m` to control the active runner object.

    Use \x1b[31m%run file.py\x1b[0m to run a python file.

            '''
            prompt_config = cfg.PromptManager
            prompt_config.in_template  = 'convis >>> <\\#>: '
            prompt_config.in2_template = '   .\\D.: '
            prompt_config.out_template = '       <<< <\\#>: '
            IPython.start_ipython(argv=[],user_ns=namespace,config=cfg,display_banner=True)
        except:
            raise
            import code
            code.InteractiveConsole(locals=namespace).interact()
    if arg == 'python':
        import code
        code.InteractiveConsole(locals=namespace).interact()

for i,a in enumerate(sys.argv):
    process_argument(a,i=i,argv=sys.argv)

print('=='*10)
print(namespace.get("model",None))
print('=='*10)