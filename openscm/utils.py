"""Utility functions for openscm.
"""

import warnings


import functools


def counted(base_func):
    """Count how many times a function has been called
    """
    # a good idea according to
    # https://realpython.com/primer-on-python-decorators/#simple-decorators...
    @functools.wraps(base_func)
    def wrapped(*args, **kwargs):
        wrapped.calls += 1

        return base_func(*args, **kwargs)

    wrapped.calls = 0

    return wrapped


@counted
def ensure_input_is_tuple(inp):
    if isinstance(inp, str):
        if ensure_input_is_tuple.calls == 1:
            warnings.warn("Converting input {} from string to tuple".format(inp))
        return (inp,)
    else:
        return inp
