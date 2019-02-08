"""Utility functions for openscm.
"""

import warnings


def ensure_input_is_tuple(inp):
    if isinstance(inp, str):
        if getattr(ensure_input_is_tuple, 'calls', 0) == 0:
            ensure_input_is_tuple.calls = 1
            warnings.warn("Converting input {} from string to tuple".format(inp))
        return (inp,)
    else:
        return inp
