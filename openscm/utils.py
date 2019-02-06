"""Utility functions for openscm.
"""
import datetime
import warnings


def ensure_input_is_tuple(inp):
    if isinstance(inp, str):
        if getattr(ensure_input_is_tuple, "calls", 0) == 0:
            ensure_input_is_tuple.calls = 1
            warnings.warn("Converting input {} from string to tuple".format(inp))
        return (inp,)
    else:
        return inp


def convert_datetime_to_openscm_time(dt_in: datetime.datetime) -> int:
    """Convert a datetime.datetime instance to OpenSCM time i.e. seconds since 1970-1-1 00:00:00"""
    openscm_reference_time = datetime.datetime(1970, 1, 1, 0, 0, 0)

    return int((dt_in - openscm_reference_time).total_seconds())
