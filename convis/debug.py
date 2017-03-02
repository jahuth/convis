import datetime


do_debug = False
do_replace_inputs = True

debug_messages = []

def send_dbg(channel,msg,log_level=0):
    global debug_messages
    debug_messages.append((channel,msg,log_level,datetime.datetime.now()))

def _filter_dbg_txt(search,txt):
    if search is None:
        return True
    if type(search) is list:
        for s in search:
            if not _filter_dbg_txt(s,txt):
                return False    
        return True
    return search in txt

def filter_dbg(channel=None,msg=None,log_level=0):
    global debug_messages
    return [d for d in debug_messages[::-1] if _filter_dbg_txt(channel,d[0]) and _filter_dbg_txt(msg,d[1]) and log_level <= d[2]]
