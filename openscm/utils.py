"""
Utility functions for openscm.
"""
import datetime
import warnings
from typing import Any, Tuple, Union

from dateutil.relativedelta import relativedelta

OPENSCM_REFERENCE_TIME = datetime.datetime(1970, 1, 1, 0, 0, 0)


def ensure_input_is_tuple(inp: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """
    Return parameter as a tuple.

    Parameters
    ----------
    inp
        String or tuple to return as a tuple

    Returns
    -------
    Tuple[str, ...]
        A tuple with a single string `inp` if `inp` is a string, otherwise return `inp`
    """
    if isinstance(inp, str):
        if not getattr(ensure_input_is_tuple, "calls", 0):
            setattr(ensure_input_is_tuple, "calls", 1)
            warnings.warn("Converting input {} from string to tuple".format(inp))
        return (inp,)

    return inp


def convert_datetime_to_openscm_time(dt_in: datetime.datetime) -> int:
    """Convert a datetime.datetime instance to OpenSCM time i.e. seconds since OPENSCM_REFERENCE_TIME"""
    return int((dt_in - OPENSCM_REFERENCE_TIME).total_seconds())


def convert_openscm_time_to_datetime(oscm_in: int) -> datetime.datetime:
    """Convert OpenSCM time to datetime.datetime"""
    return OPENSCM_REFERENCE_TIME + relativedelta(seconds=oscm_in)


def is_floatlike(f: Any) -> bool:
    """
    Check if input can be cast to a float

    This includes strings such as "6.03" which can be cast to a float

    Parameters
    ----------
    f
        Input

    Returns
    -------
        True if f can be cast to a float
    """
    if isinstance(f, (int, float)):
        return True

    try:
        float(f)
        return True
    except (TypeError, ValueError):
        return False
