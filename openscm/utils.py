"""
Utility functions for openscm.
"""

from typing import Tuple, Union
import warnings


def ensure_input_is_tuple(inp: Union[str, Tuple[str]]) -> Tuple[str]:
    """
    Return parameter as a tuple.

    Parameters
    ----------
    inp
        String or tuple to return as a tuple

    Returns
    -------
    A tuple with a single string `inp` if `inp` is a string, otherwise return `inp`
    """
    if isinstance(inp, str):
        if getattr(ensure_input_is_tuple, "calls", 0) == 0:
            ensure_input_is_tuple.calls = 1
            warnings.warn("Converting input {} from string to tuple".format(inp))
        return (inp,)

    return inp
