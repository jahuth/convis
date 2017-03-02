def unique_list(seq):
    """ This function is removing duplicates from a list while keeping the order """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

from contextlib import contextmanager
 
@contextmanager
def suppress(*exceptions):
    try:
        yield
    except exceptions:
        pass
