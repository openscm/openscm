"""
A simplified version of pandas DateOffset's which use datetime-like objects instead of pd.Timestamp.

This differentiation allows for time's which exceed the range of pd.Timestamp which is particularly important for longer running
models
"""
import functools
from datetime import datetime

from pandas.tseries.frequencies import to_offset as pd_to_offset
from pandas.tseries.offsets import (
    BusinessMixin,
    NaT,
    as_datetime,
    conversion,
    normalize_date,
)


def apply_dt(func, self):
    """
    A simplified version of wrapper pandas.tseries.offsets.apply_wraps which keeps the result as a datetime instead of converting to
    pd.Timestamp
    This the pandas Offset object's apply method has to be manually updated
    """

    @functools.wraps(func)
    def wrapper(other: datetime) -> datetime:
        if other is NaT:
            return NaT

        tz = getattr(other, "tzinfo", None)

        result = func(self, as_datetime(other))

        if self.normalize:
            # normalize_date returns normal datetime
            result = normalize_date(result)

        if tz is not None and result.tzinfo is None:
            result = conversion.localize_pydatetime(result, tz)

        return result

    return wrapper


def apply_rollforward(self):
    """
    Roll provided date backward to next offset only if not on offset.
    """

    def wrapper(dt: datetime) -> datetime:
        dt = as_datetime(dt)
        if not self.onOffset(dt):
            dt = dt + self.__class__(1, normalize=self.normalize, **self.kwds)
        return dt

    return wrapper


def apply_rollback(self):
    """
    Roll provided date backward to next offset only if not on offset.
    """

    def wrapper(dt: datetime) -> datetime:
        dt = as_datetime(dt)
        if not self.onOffset(dt):
            dt = dt - self.__class__(1, normalize=self.normalize, **self.kwds)
        return dt

    return wrapper


def to_offset(rule):
    """
    Takes a rule and returns a DateOffset class

    The Offset class is manipulated to return datetimes instead of pd.Timestamps

    Parameters
    ----------
    rule:

    Returns
    -------
    DateOffset
    """
    offset = pd_to_offset(rule)

    if isinstance(offset, BusinessMixin) or offset.rule_code.startswith("B"):
        raise ValueError(
            "Invalid rule for offset - Business related offsets are not supported"
        )

    def wrap_funcs(fname):
        # Checks if the function has been wrapped and replace with `apply_dt` wrapper
        func = getattr(offset, fname)

        if hasattr(func, "__wrapped__"):
            orig_func = func.__wrapped__
            object.__setattr__(offset, fname, apply_dt(orig_func, offset))

    wrap_funcs("apply")
    object.__setattr__(offset, "rollforward", apply_rollforward(offset))
    object.__setattr__(offset, "rollback", apply_rollback(offset))

    return offset


def generate_range(start: datetime, end: datetime, offset):
    """
    Generates a range of datetime objects between start and end, using offset to determine the steps

    The range will extend either end of the span to the next valid timestep. For example with a start value of 2001-04-01 and
    a YearStart offsetter, the first value from the generator will be 2001-01-01.

    Parameters
    ----------
    start: datetime
        Starting datetime
    end: datetime
        End datetime. Values upto and including this value are returned
    offset: DateOffset
        Offset object for determining the timesteps. An offsetter obtained from `openscm.scmdataframe.offset.to_offset` *must* be
        used.
    Returns
    -------
    datetime generator
    """
    # Get the bounds
    start = offset.rollback(start)
    end = offset.rollforward(end)

    # Iterate to find all the required timesteps
    current = start
    while current <= end:
        yield current

        next_current = offset.apply(current)
        if next_current <= current:
            raise ValueError(
                "Offset is not increasing datetime: {}".format(current.isotime())
            )

        current = next_current
