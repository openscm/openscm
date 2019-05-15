"""
Utility functions for openscm.
"""
import datetime
from typing import Any, Sequence, Union

from dateutil.relativedelta import relativedelta

HierarchicalName = Union[str, Sequence[str]]

# TODO get rid of:
OPENSCM_REFERENCE_TIME = datetime.datetime(1970, 1, 1, 0, 0, 0)


def hierarchical_name_as_sequence(inp: HierarchicalName) -> Sequence[str]:
    """
    TODO Return parameter as a tuple.

    Parameters
    ----------
    inp
        String or tuple to return as a tuple

    Returns
    -------
    Sequence[str]
        A sequence with a single string `inp` if `inp` is a string, otherwise return
        `inp`
    """
    if isinstance(inp, str):
        return inp.split("|")

    return inp


# TODO get rid of:
def convert_datetime_to_openscm_time(dt_in: datetime.datetime) -> int:
    """Convert a datetime.datetime instance to OpenSCM time i.e. seconds since OPENSCM_REFERENCE_TIME"""
    return int((dt_in - OPENSCM_REFERENCE_TIME).total_seconds())


# TODO get rid of:
def convert_openscm_time_to_datetime(oscm_in: int) -> datetime.datetime:
    """Convert OpenSCM time to datetime.datetime"""
    # Need to cast to int as np.int64 from numpy arrays are unsupported
    return OPENSCM_REFERENCE_TIME + relativedelta(seconds=int(oscm_in))


def is_floatlike(f: Any) -> bool:
    """
    Check if input can be cast to a float.

    This includes strings such as "6.03" which can be cast to a float

    Parameters
    ----------
    f
        Input

    Returns
    -------
    bool
        True if f can be cast to a float
    """
    if isinstance(f, (int, float)):
        return True

    try:
        float(f)
        return True
    except (TypeError, ValueError):
        return False
