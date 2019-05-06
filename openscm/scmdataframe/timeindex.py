"""
OpenSCM's time index, used for ``ScmDataFrameBase``, and helper functions.
"""
import datetime
from typing import Any, List

import numpy as np
import pandas as pd
from dateutil import parser

from openscm.utils import (
    convert_datetime_to_openscm_time,
    convert_openscm_time_to_datetime,
    is_floatlike,
)


# pylint doesn't recognise return statements if they include 'of' but it should, see
# https://github.com/PyCQA/pylint/pull/2884 and search for ':obj:`list` of :obj:`str`'
# in https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
def to_int(x: np.ndarray) -> np.ndarray:  # pylint: disable=missing-return-doc
    """
    Convert inputs to int and check conversion is sensible

    Parameters
    ----------
    x
        Values to convert

    Returns
    -------
    :obj:`np.array` of :obj:`int`
        Input, converted to int

    Raises
    ------
    ValueError
        If the int representation of any of the values is not equal to its original
        representation (where equality is checked using the ``!=`` operator).

    TypeError
        x is not a ``np.ndarray``
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(
            "For our own sanity, this method only works with np.ndarray input. "
            "x is type: {}".format(type(x))
        )
    cols = np.array([int(v) for v in x])
    invalid_vals = x[cols != x]
    if invalid_vals.size:
        raise ValueError("invalid values `{}`".format(list(invalid_vals)))

    return cols


def npdt64_to_datetime(dt: np.datetime64) -> datetime.datetime:
    """
    Convert a ``numpy.datetime64`` instance to a ``datetime.datetime``

    Parameters
    ----------
    dt
        Value to convert

    Returns
    -------
    :obj:`datetime.datetime`
        ``datetime.datetime`` equivalent of ``dt``
    """
    # pandas method doesn't contain type hint so mypy isn't happy
    return pd.Timestamp(dt).to_pydatetime()  # type: ignore


# pylint doesn't recognise return statements if they include 'of' but it should, see
# https://github.com/PyCQA/pylint/pull/2884 and search for ':obj:`list` of :obj:`str`'
# in https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
def _format_datetime(  # pylint: disable=missing-return-doc
    dts: List[Any]
) -> List[datetime.datetime]:
    """
    Convert a list into a set of ``datetime.datetime``'s

    Parameters
    ----------
    dts
        Input to attempt to convert

    Returns
    -------
    :obj:`list` of :obj:`datetime.datetime`
        Converted ``datetime.datetime``'s

    Raises
    ------
    ValueError
        If one of the values in ``dts`` cannot be converted to ``datetime.datetime``
    """
    empty_input = not dts.size if isinstance(dts, np.ndarray) else not dts
    if empty_input:
        return []
    dt_0 = dts[0]

    if isinstance(dt_0, (int, np.int64)):
        # Year strings
        dts = [datetime.datetime(y, 1, 1) for y in to_int(dts)]
    elif isinstance(dt_0, np.datetime64):
        dts = [npdt64_to_datetime(dt) for dt in dts]
    elif is_floatlike(dt_0):

        def convert_float_to_datetime(inp):
            year = int(inp)
            fractional_part = inp - year
            base = datetime.datetime(year, 1, 1)
            return base + datetime.timedelta(
                seconds=(base.replace(year=year + 1) - base).total_seconds()
                * fractional_part
            )

        dts = [convert_float_to_datetime(float(t)) for t in dts]
    elif isinstance(dt_0, str):
        try:
            dts = [parser.parse(dt) for dt in dts]
        except ValueError:
            pass  # can't convert, catch lower down
    elif isinstance(dt_0, pd.Timestamp):
        dts = [dt.to_pydatetime() for dt in dts]

    not_datetime = [not isinstance(x, datetime.datetime) for x in dts]
    if any(not_datetime):
        bad_values = np.asarray(dts)[not_datetime]
        error_msg = (
            "All time values must be convertible to datetime. The following "
            "values are not:\n\t{}".format(bad_values)
        )
        raise ValueError(error_msg)

    return dts


class TimeIndex:
    """
    Keeps track of both datetime and openscm datetimes and knows how to convert between the two formats
    """

    def __init__(self, py_dt=None, openscm_dt=None):
        if not (py_dt is not None or openscm_dt is not None):
            raise AssertionError("One of `py_dt` or `openscm_dt` must be supplied")

        if py_dt is not None:
            py_dt = _format_datetime(np.asarray(py_dt))
            object.__setattr__(self, "_py", np.asarray(py_dt))
            object.__setattr__(
                self,
                "_openscm",
                np.asarray([convert_datetime_to_openscm_time(dt) for dt in py_dt]),
            )
        else:
            object.__setattr__(
                self,
                "_py",
                np.asarray([convert_openscm_time_to_datetime(dt) for dt in openscm_dt]),
            )
            object.__setattr__(self, "_openscm", np.asarray(openscm_dt))

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Set an attribute of self

        Parameters
        ----------
        key
            Attribute to set

        value
            Value to set

        Raises
        ------
        AttributeError
            ``TimeIndex`` is immutable and hence no attributes can be set.
        """
        raise AttributeError("TimeIndex is immutable")

    # pylint doesn't recognise return statements if they include 'of' but it should,
    # see https://github.com/PyCQA/pylint/pull/2884 and search for ':obj:`list` of
    # :obj:`str`' in
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    def as_openscm(self) -> np.ndarray:  # pylint: disable=missing-return-doc
        """
        Get time points as OpenSCM times

        For details of how to convert between datetime.datetime and OpenSCM time, see
        docstring of ``convert_datetime_to_openscm_time``.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Datetime representation of each time point
        """
        # mypy and pylint aren't smart enough to work out ``._py`` attribute exists
        return self._openscm  # type: ignore # pylint: disable=no-member

    # pylint doesn't recognise return statements if they include 'of' but it should,
    # see https://github.com/PyCQA/pylint/pull/2884 and search for ':obj:`list` of
    # :obj:`str`' in
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    def as_py(self) -> np.ndarray:  # pylint: disable=missing-return-doc
        """
        Get time points as Python datetimes

        Returns
        -------
        :obj:`np.array` of :obj:`datetime.datetime`
            Datetime representation of each time point
        """
        # mypy and pylint aren't smart enough to work out ``._py`` attribute exists
        return self._py  # type: ignore # pylint: disable=no-member

    def as_pd_index(self) -> pd.Index:
        """
        Get time points as pd.Index

        Returns
        -------
        :obj:`pd.Index`
            pd.Index of dtype "object" with name "time" made from the time points
        """
        return pd.Index(self.as_py(), dtype="object", name="time")

    # pylint doesn't recognise return statements if they include 'of' but it should,
    # see https://github.com/PyCQA/pylint/pull/2884 and search for ':obj:`list` of
    # :obj:`str`' in
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    def years(self) -> np.ndarray:  # pylint: disable=missing-return-doc
        """
        Get year of each time point

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Year of each time point
        """
        return np.array([dt.year for dt in self.as_py()])

    # pylint doesn't recognise return statements if they include 'of' but it should,
    # see https://github.com/PyCQA/pylint/pull/2884 and search for ':obj:`list` of
    # :obj:`str`' in
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    def months(self) -> np.ndarray:  # pylint: disable=missing-return-doc
        """
        Get month of each time point

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Month of each time point
        """
        return np.array([dt.month for dt in self.as_py()])

    # pylint doesn't recognise return statements if they include 'of' but it should,
    # see https://github.com/PyCQA/pylint/pull/2884 and search for ':obj:`list` of
    # :obj:`str`' in
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    def days(self) -> np.ndarray:  # pylint: disable=missing-return-doc
        """
        Get day of each time point

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Day of each time point
        """
        return np.array([dt.day for dt in self.as_py()])

    # pylint doesn't recognise return statements if they include 'of' but it should,
    # see https://github.com/PyCQA/pylint/pull/2884 and search for ':obj:`list` of
    # :obj:`str`' in
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    def hours(self) -> np.ndarray:  # pylint: disable=missing-return-doc
        """
        Get hour of each time point

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Hour of each time point
        """
        return np.array([dt.hour for dt in self.as_py()])

    # pylint doesn't recognise return statements if they include 'of' but it should,
    # see https://github.com/PyCQA/pylint/pull/2884 and search for ':obj:`list` of
    # :obj:`str`' in
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    def weekdays(self) -> np.ndarray:  # pylint: disable=missing-return-doc
        """
        Get weekday of each time point

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Day of the week of each time point
        """
        return np.array([dt.weekday() for dt in self.as_py()])
