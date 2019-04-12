from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import parser

from openscm.utils import convert_datetime_to_openscm_time, convert_openscm_time_to_datetime, is_floatlike


def to_int(x):
    """Formatting series or timeseries columns to int and checking validity.
    If `index=False`, the function works on the `pd.Series x`; else,
    the function casts the index of `x` to int and returns x with a new index.
    """
    cols = list(map(int, x))
    error = x[cols != x]
    if len(error):
        raise ValueError('invalid values `{}`'.format(list(error)))
    return cols


def npdt_to_datetime(dt):
    return pd.Timestamp(dt).to_pydatetime()

def _format_datetime(dts):
    if not len(dts):
        return []
    dt_0 = dts[0]

    if isinstance(dt_0, (int, np.int64)):
        # Year strings
        dts = [datetime(y, 1, 1) for y in to_int(dts)]
    elif isinstance(dt_0, np.datetime64):
        dts = [pd.Timestamp(dt).to_pydatetime() for dt in dts]
    elif is_floatlike(dt_0):
        def convert_float_to_datetime(inp):
            year = int(inp)
            fractional_part = inp - year
            base = datetime(year, 1, 1)
            return base + timedelta(
                seconds=(base.replace(year=year + 1) - base).total_seconds()
                        * fractional_part
            )

        dts = [convert_float_to_datetime(float(t)) for t in dts]
    elif isinstance(dt_0, str):
        dts = [parser.parse(dt) for dt in dts]
    elif isinstance(dt_0, pd.Timestamp):
        dts = [dt.to_pydatetime() for dt in dts]

    not_datetime = [
        not isinstance(x, datetime) for x in dts
    ]
    if any(not_datetime):
        bad_values = np.asarray(dts)[not_datetime]
        error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(
            bad_values
        )
        raise ValueError(error_msg)

    return dts


class TimeIndex(object):
    """
    Keeps track of both datetime and openscm datetimes and knows how to convert between the two formats
    """

    def __init__(self, py_dt=None, openscm_dt=None):
        assert py_dt is not None or openscm_dt is not None, "Can only pass either python datetimes or openscm datetimes"
        if py_dt is not None:
            py_dt = _format_datetime(np.asarray(py_dt))
            object.__setattr__(self, '_py', np.asarray(py_dt))
            object.__setattr__(self, '_openscm', np.asarray([convert_datetime_to_openscm_time(dt) for dt in py_dt]))
        else:
            object.__setattr__(self, '_py', np.asarray([convert_openscm_time_to_datetime(dt) for dt in openscm_dt]))
            object.__setattr__(self, '_openscm', np.asarray(openscm_dt))

    def __setattr__(self, key, value):
        raise AttributeError('TimeIndex is immutable')

    def as_openscm(self):
        return self._openscm

    def as_py(self):
        return self._py

    def as_pd_index(self):
        return pd.Index(self._py, dtype='object', name='time')

    def years(self):
        return np.array([dt.year for dt in self._py])

    def months(self):
        return np.array([dt.month for dt in self._py])

    def days(self):
        return np.array([dt.day for dt in self._py])

    def hours(self):
        return np.array([dt.hour for dt in self._py])

    def weekdays(self):
        return np.array([dt.weekday() for dt in self._py])
