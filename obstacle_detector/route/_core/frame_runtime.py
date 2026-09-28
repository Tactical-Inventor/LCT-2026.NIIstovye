"""Keep unrelated cyclic garbage collection outside bounded frame computations.

These routines allocate acyclic NumPy arrays/dicts. Reference-count reclamation
continues normally. The caller's cyclic-GC state is restored, including errors.
This is not a hard real-time guarantee against OS scheduling pauses.
"""
import gc
from functools import wraps


def frame_compute(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        enabled=gc.isenabled()
        if enabled:gc.disable()
        try:
            return function(*args,**kwargs)
        finally:
            if enabled:gc.enable()
    return wrapped
