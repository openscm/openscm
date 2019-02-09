"""Utility functions for openscm.
"""
import datetime
from dateutil import relativedelta
import warnings


OPENSCM_REFERENCE_TIME = datetime.datetime(1970, 1, 1, 0, 0, 0)


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
    return int((dt_in - OPENSCM_REFERENCE_TIME).total_seconds())


def convert_openscm_time_to_datetime(oscm_in: int) -> datetime.datetime:
    """Convert OpenSCM time to datetime.datetime"""
    return OPENSCM_REFERENCE_TIME + relativedelta.relativedelta(seconds=oscm_in)


def round_to_nearest_year(dtin):
        """Round a datetime to Jan 1st 00:00:00 of the nearest year

        thank you https://stackoverflow.com/a/48108115"""
        dt_start_year = dtin.replace(
            month=1, day=1, minute=0, hour=0, second=0, microsecond=0
        )
        dt_half_year = dtin.replace(month=6, day=17)
        if dtin > dt_half_year:
            return dt_start_year + relativedelta.relativedelta(years=1)
        else:
            return dt_start_year
