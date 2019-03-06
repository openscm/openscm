"""
Utility functions for openscm.
"""

from typing import Tuple, Union
import warnings

_ensure_input_is_tuple_calls: int = 0


def ensure_input_is_tuple(inp: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
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
    global _ensure_input_is_tuple_calls  # pylint: disable=global-statement
    if isinstance(inp, str):
        if not _ensure_input_is_tuple_calls:
            _ensure_input_is_tuple_calls = 1
            warnings.warn("Converting input {} from string to tuple".format(inp))
        return (inp,)

    return inp
