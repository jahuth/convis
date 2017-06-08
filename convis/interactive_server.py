import numpy as np
import matplotlib
import matplotlib.pylab as plt

namespace = {}

template = """
<html><head>
<script src="https://cdn.jsdelivr.net/jquery/3.2.1/jquery.min.js"></script>
<script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
<style>
body {
    font-family: sans-serif;
}
#namespace_list_div {
    border: 5px solid #888;
    background-color: #aaa;
    margin: 20px;
    padding:10px;
    width: 300px;
    float: left;
}

ul#namespace_list {
    list-style-type: none;
    margin: 0;
    padding: 0;
}

ul#namespace_list a {
    color: #444;
}
ul#namespace_list a:hover {
    color: black;
}

#disp {
    background-color: #efefef;
    margin: 20px;
    padding: 10px;
}

#pinned_bar div {
    margin:2px;
    padding: 2px;
    background-color: #efdead;
    float: left;
}
</style>
<script>

var loading = "loading...";

function feedback(data) {
    $('#feedback').html(data).finish().show( "slow" ).animate({ opacity: 1.0 }, 1500 ).animate({ opacity: 0.4 }, 1500 ).hide( "slow" );
}

function init() {
    $('#pinned_bar').sortable({ revert: true });
    update(); // start poll loop
}
var selected = '';
namepsace = [];
function show(u) {
    selected = '';
    var start_time = new Date().getTime();
    $('#disp').html(loading);
    $.get('/show/'+u, function (data) {
        $('#disp').html(data);
        $('#disp').find('select.namespace_select').each(function (i) {
            var ns_elem = $(this);
            $.each(namespace, function (i, elm) {
                var o = ns_elem.append("<option>"+elm+"</option>")
            });
            ns_elem.change(function() {
                set($(this).data('slot'),$(this).val());
            });
        });
        $('#disp').find('img.mjpg').each(function (i) {
            $(this).click(function () {
                $('<div><img src="'+$(this).attr('src')+'"></div>').appendTo('#pinned_bar').draggable({
                  connectToSortable: "#pinned_bar",
                  containment: $("#pinned_bar"),
                  revert: "invalid"
                }).dblclick(function () {
                    $(this).remove();
                });
            });
        });
        if (new Date().getTime() - start_time < 500) {
            // fast requests can be repeated for updates
            selected = u;
        }
    });
};

function describe(u) {
    selected = '';
    var start_time = new Date().getTime();
    $('#disp').html(loading);
    $.get('/describe/'+u, function (data) {
        $('#disp').html(data);
        $('#disp').find('select.namespace_select').each(function (i) {
            var ns_elem = $(this);
            $.each(namespace, function (i, elm) {
                var o = ns_elem.append("<option>"+elm+"</option>")
            });
            ns_elem.change(function() {
                set($(this).data('slot'),$(this).val());
            });
        });
        $('#disp').find('img.mjpg').each(function (i) {
            $(this).click(function () {
                $('<div><img src="'+$(this).attr('src')+'"></div>').appendTo('#pinned_bar').draggable({
                  connectToSortable: "#pinned_bar",
                  containment: $("#pinned_bar"),
                  revert: "invalid"
                }).dblclick(function () {
                    $(this).remove();
                });
            });
        });
        if (new Date().getTime() - start_time < 500) {
            // fast requests can be repeated for updates
            selected = u;
        }
    });
};


function action(slot) {
    $.get('/action/'+selected+'/'+slot, function (data) {
        feedback(data);
    });
    show(selected);
}

function set(slot, value) {
    $.get('/set/'+selected+'/'+slot+'/'+value, function (data) {
        feedback(data);
    });
    show(selected);
}


function fetch(u) {
    // deprecated?
    $('#disp').html(loading);
    $.get('/show/'+u, function (data) {
        $('#disp').html(data);
        $('#disp').find('select.namespace_select').each(function (i) {
            var ns_elem = $(this);
            $.each(namespace, function (i, elm) {
                var o = ns_elem.append("<option>"+elm+"</option>")
            });
            ns_elem.change(function() {
                set($(this).data('slot'),$(this).val());
            });
        });
    });
};

function status(u) {
    //$('#disp_status').append(loading);
    $.get('/status/'+u, function (data) {
        $('#disp_status').html(data);
    });
};

function update() {
    $.ajax({
        url:'/namespace',
        success: function (data) {
            namespace = data;
            var s = '';
            $.each(data, function (i, elm) {
                s = s + "<li><a href=\\\"javascript:show('"+elm+"');\\\">"+elm+"</a></li>";
            });        
            $('#namespace_list').html(s);
        },
        dataType: "json" 
    });
    status(selected);
    window.setTimeout(update,500)
}
</script>
<title>convis web interface</title>
</head>

<body onload='init();'>
<div id="feedback" style="display:none; position:absolute; top: 10px; border: 5px solid gray; background-color:#efefef;"></div>
<div id='pinned_bar' style='height: 100px;'></div>

<div id='namespace_list_div'>
<ul id='namespace_list'>
</ul>
</div>

<div id='disp'></div>
<div id='disp_status'></div>

</body>
</html>
"""

closing_down = False

def start_server():
    global namespace
    import SocketServer
    import SimpleHTTPServer
    import inspect
    import os
    PORT = 8080 + np.random.randint(100)

    def move():
        """ sample function to be called via a URL"""
        return 'hi'

    class CustomHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
        logging = False
        promises = []
        def log_message(self, format, *args):
            pass
        def do_GET(self):
            if self.path=='/':
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()
                with open(os.path.join(os.path.dirname(__file__), 'html_template.html')) as f:
                    template = f.read()
                self.wfile.write(template)
            elif self.path == '/namespace':
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()
                s = []
                for k,v in namespace.items():
                    if (not k.startswith('_') 
                        and 
                            not k.endswith('_') 
                        and 
                            (hasattr(v,'_repr_html_') 
                                or hasattr(v,'_repr_html_in_namespace_') 
                                or hasattr(v,'_web_repr_') 
                                or hasattr(v,'_web_repr_status_') 
                                or hasattr(v,'__html__') 
                                or type(v) == matplotlib.figure.Figure
                                or hasattr(v,'__array__'))
                        and
                            not inspect.isclass(v)
                        and
                            not str(type(v)) in ['<type \'module\'>','<type \'instancemethod\'>']
                        ):
                        s.append('"'+str(k)+'"')
                self.wfile.write('['+','.join(s)+']')
                return
            elif self.path.startswith('/mjpg/'):
                k = self.path.split('/')[2]
                if 'output' in namespace.keys():
                    v = namespace['output']
                if k in namespace.keys():
                    v = namespace[k]
                if hasattr(v,'get_image'):
                    try:
                        from PIL import Image
                    except:
                        import Image
                    import StringIO
                    import time
                    self.send_response(200)
                    self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
                    self.end_headers()
                    while closing_down is False:
                        try:
                            img = v.get_image()
                            if np.max(img) <= 1.0:
                                img = img * 256.0
                            if np.max(img) > 256.0:
                                img = img/np.max(img) * 256.0
                            jpg = Image.fromarray(img).convert('RGB')
                            tmpFile = StringIO.StringIO()
                            jpg.save(tmpFile,'JPEG')
                            self.wfile.write("--jpgboundary")
                            self.send_header('Content-type','image/jpeg')
                            self.send_header('Content-length',str(tmpFile.len))
                            self.end_headers()
                            jpg.save(self.wfile,'JPEG')
                            time.sleep(0.05)
                        except Exception as e:
                            img = np.zeros((50,50))
                            for i in range(img.shape[0]):
                                img[i,i] = 255.0
                                img[i,-i] = 255.0
                            jpg = Image.fromarray(img).convert('RGB')
                            tmpFile = StringIO.StringIO()
                            jpg.save(tmpFile,'JPEG')
                            self.wfile.write("--jpgboundary")
                            self.send_header('Content-type','image/jpeg')
                            self.send_header('Content-length',str(tmpFile.len))
                            self.end_headers()
                            jpg.save(self.wfile,'JPEG')
                            print e
                            break
                else:
                    self.send_response(404)
                    self.send_header('Content-type','text/html')
                    self.end_headers()     
                    self.wfile.write("no stream found")               
                return
            elif self.path.startswith('/status/'):
                k = self.path[8:]
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()
                v = namespace.get(k,None)
                if v is not None:
                    if type(v) == matplotlib.figure.Figure:
                        import StringIO, urllib, base64
                        imgdata = StringIO.StringIO()
                        v.savefig(imgdata, format='png')
                        imgdata.seek(0) 
                        image = base64.encodestring(imgdata.buf) 
                        self.wfile.write("<img src='data:image/png;base64," + urllib.quote(image) + "'>")
                    elif hasattr(v,'_web_repr_status_'):
                        self.wfile.write(v._web_repr_status_(name=k,namespace=namespace))
                    elif hasattr(v,'__array__'):
                        self.wfile.write('<pre>'+str(v)+'</pre>')
                return
            elif self.path.startswith('/show/'):
                k = self.path[6:]
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()
                v = namespace.get(k,None)
                if v is not None:
                    if type(v) == matplotlib.figure.Figure:
                        self.wfile.write("Matplotlib Figure")
                    elif hasattr(v,'_web_repr_'):
                        self.wfile.write(v._web_repr_(name=k,namespace=namespace))
                    elif hasattr(v,'_repr_html_'):
                        try:
                            self.wfile.write(v._repr_html_(namespace=namespace))
                        except:
                            self.wfile.write(v._repr_html_())
                    elif hasattr(v,'__html__'):
                        self.wfile.write(v.__html__())
                    elif hasattr(v,'__array__'):
                        self.wfile.write('Numpy Array '+str(v.shape))
                        self.wfile.write('Mean: '+str(v.mean())+'<br\>')
                        self.wfile.write('Max: '+str(v.max())+'<br\>')
                        self.wfile.write('Min: '+str(v.min())+'<br\>')
                        try:
                            from convis import variable_describe
                            promid = len(self.promises)
                            self.promises.append(lambda: variable_describe._tensor_to_html(v))
                            self.wfile.write('<div class="promise" promise="'+str(promid)+'"></div>')
                        except:
                            self.wfile.write('<div>could not give promise</div>')
                    else:
                        self.wfile.write('unrecognized type: '+str(k)+'.')
                return
            elif self.path.startswith('/promise/'):
                k = self.path[9:]
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()
                try:
                    self.wfile.write(self.promises[int(k)]())
                except:
                    self.wfile.write('An error occured.')
                    raise
            elif self.path.startswith('/set/'):
                p = self.path.split('/')
                k = p[2]
                slot = p[3]
                value = p[4]
                v = namespace.get(k,None)
                if hasattr(v,'_web_set_'):
                    self.send_response(200)
                    self.send_header('Content-type','text/html')
                    self.end_headers()
                    self.wfile.write(v._web_set_(slot, value, namespace=namespace))
                    return
            elif self.path.startswith('/action/'):
                p = self.path.split('/')
                k = p[2]
                act = p[3]
                v = namespace.get(k,None)
                if hasattr(v,'_web_action_'):
                    self.send_response(200)
                    self.send_header('Content-type','text/html')
                    self.end_headers()
                    self.wfile.write(v._web_action_(act, namespace=namespace))
                    return
            elif self.path=='/move':
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()
                self.wfile.write(move())
                return
            else:
                self.send_response(404)
                self.send_header('Content-type','text/html')
                self.end_headers()
                self.wfile.write("Not a valid request!")
                pass #SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
    httpd = SocketServer.ThreadingTCPServer(('localhost', PORT),CustomHandler)
    print "serving at port", PORT
    import webbrowser
    webbrowser.open('http://localhost:'+str(PORT))
    httpd.serve_forever()