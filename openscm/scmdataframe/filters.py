"""
Helpers for filtering DataFrames.

Borrowed from :obj:`pyam.utils`.
"""

import datetime
import re
import time
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from ..core.parameters import HIERARCHY_SEPARATOR


def is_in(vals: List, items: List) -> np.ndarray:
    """
    Find elements of vals which are in items.

    Parameters
    ----------
    vals
        The list of values to check

    items
        The options used to determine whether each element of :obj:`vals` is in the
        desired subset or not

    Returns
    -------
    :obj:`np.array` of :obj:`bool`
        Array of the same length as :obj:`vals` where the element is ``True`` if the
        corresponding element of :obj:`vals` is in :obj:`items` and False otherwise
    """
    return np.array([v in items for v in vals])


def find_depth(
    meta_col: pd.Series,
    s: str,
    level: Union[int, str],
    separator: str = HIERARCHY_SEPARATOR,
) -> np.ndarray:
    """
    Find all values which match given depth from a filter keyword.

    Parameters
    ----------
    meta_col
        Column in which to find values which match the given depth

    s
        Filter keyword, from which level should be applied

    level
        Depth of value to match as defined by the number of separator in the value name.
        If an int, the depth is matched exactly. If a str, then the depth can be matched
        as either "X-", for all levels up to level "X", or "X+", for all levels above
        level "X".

    separator
        The string used to separate levels in s. Defaults to a pipe ("|").

    Returns
    -------
    :obj:`np.array` of :obj:`bool`
        Array where ``True`` indicates a match

    Raises
    ------
    ValueError
        If :obj:`level` cannot be understood
    """
    # determine function for finding depth level
    if not isinstance(level, str):

        def test(x):
            return level == x

    elif level[-1] == "-":
        _level = int(level[:-1])

        def test(x):
            return _level >= x

    elif level[-1] == "+":
        _level = int(level[:-1])

        def test(x):
            return _level <= x

    else:
        raise ValueError("Unknown level type: {}".format(level))

    # determine depth
    pipe = re.compile(re.escape(separator))
    regexp = str(s).replace("*", "")

    def apply_test(val):
        return test(len(pipe.findall(val.replace(regexp, ""))))

    return np.array([b for b in [apply_test(m) for m in meta_col]])


def pattern_match(  # pylint: disable=too-many-arguments,too-many-locals
    meta_col: pd.Series,
    values: Union[Iterable[str], str],
    level: Optional[Union[str, int]] = None,
    regexp: bool = False,
    has_nan: bool = True,
    separator: str = HIERARCHY_SEPARATOR,
) -> np.ndarray:
    """
    Filter data by matching metadata columns to given patterns.

    Parameters
    ----------
    meta_col
        Column to perform filtering on

    values
        Values to match

    level
        Passed to :func:`find_depth`. For usage, see docstring of :func:`find_depth`.

    regexp
        If ``True``, match using regexp rather than pseudo regexp syntax of `pyam
        <https://github.com/IAMconsortium/pyam>`_.

    has_nan
        If ``True``, convert all nan values in :obj:`meta_col` to empty string before
        applying filters. This means that "" and "*" will match rows with
        :class:`np.nan`. If ``False``, the conversion is not applied and so a search in
        a string column which contains :class:`np.nan` will result in a
        :class:`TypeError`.

    separator
        String used to separate the hierarchy levels in values. Defaults to '|'

    Returns
    -------
    :obj:`np.array` of :obj:`bool`
        Array where ``True`` indicates a match

    Raises
    ------
    TypeError
        Filtering is performed on a string metadata column which contains
        :class:`np.nan` and :obj:`has_nan` is ``False``
    """
    matches = np.array([False] * len(meta_col))
    _values = (
        [values]
        if not isinstance(values, Iterable) or isinstance(values, str)
        else values
    )

    # pyam issue (#40) with string-to-nan comparison, replace nan by empty string
    # TODO: add docs and example of filtering/removing NaN given this internal
    #       conversion
    _meta_col = meta_col.copy()
    if has_nan:
        _meta_col.loc[
            [np.isnan(i) if not isinstance(i, str) else False for i in _meta_col]
        ] = ""

    for s in _values:
        if isinstance(s, str):
            _regexp = (
                str(s)
                .replace("|", "\\|")
                .replace(".", r"\.")  # `.` has to be replaced before `*`
                .replace("*", ".*")
                .replace("+", r"\+")
                .replace("(", r"\(")
                .replace(")", r"\)")
                .replace("$", "\\$")
            ) + "$"
            pattern = re.compile(_regexp if not regexp else str(s))
            try:
                subset = [m for m in _meta_col if pattern.match(m)]
            except TypeError as e:
                # if it's not the cryptic pandas message we expect, raise
                msg = str(e)
                if msg != "expected string or bytes-like object":
                    raise e  # pragma: no cover # emergency valve

                error_msg = (
                    "String filtering cannot be performed on column '{}', which "
                    "contains NaN's, unless `has_nan` is True".format(_meta_col.name)
                )
                raise TypeError(error_msg)

            depth = (
                True
                if level is None
                else find_depth(_meta_col, str(s), level, separator=separator)
            )
            matches |= _meta_col.isin(subset) & depth
        else:
            matches |= meta_col == s

    return matches


def years_match(data: List, years: Union[List[int], int]) -> np.ndarray:
    """
    Match years in time columns for data filtering.

    Parameters
    ----------
    data
        Input data to perform filtering on

    years
        Years to match

    Returns
    -------
    :obj:`np.array` of :obj:`bool`
        Array where True indicates a match

    Raises
    ------
    TypeError
        If :obj:`years` is not :class:`int` or list of :class:`int`
    """
    years = [years] if isinstance(years, int) else years
    usable_int = (
        all(isinstance(y, int) for y in years)
        if isinstance(years, Iterable)
        else isinstance(years, int)
    )

    if not usable_int:
        error_msg = "`year` can only be filtered with ints or lists of ints"
        raise TypeError(error_msg)
    return is_in(data, years)


def month_match(
    data: List, months: Union[List[str], List[int], int, str]
) -> np.ndarray:
    """
    Match months in time columns for data filtering.

    Parameters
    ----------
    data
        Input data to perform filtering on

    months
        Months to match

    Returns
    -------
    :obj:`np.array` of :obj:`bool`
        Array where ``True`` indicates a match
    """
    return time_match(data, months, ["%b", "%B"], "tm_mon", "month")


def day_match(data: List, days: Union[List[str], List[int], int, str]) -> np.ndarray:
    """
    Match days in time columns for data filtering.

    Parameters
    ----------
    data
        Input data to perform filtering on

    days
        Days to match

    Returns
    -------
    :obj:`np.array` of :obj:`bool`
        Array where ``True`` indicates a match
    """
    return time_match(data, days, ["%a", "%A"], "tm_wday", "day")


def hour_match(data: List, hours: Union[List[int], int]) -> np.ndarray:
    """
    Match hours in time columns for data filtering.

    Parameters
    ----------
    data
        Input data to perform filtering on

    hours
        Hours to match

    Returns
    -------
    :obj:`np.array` of :obj:`bool`
        Array where ``True`` indicates a match
    """
    hours_list = [hours] if isinstance(hours, int) else hours
    return is_in(data, hours_list)


def time_match(
    data: List,
    times: Union[List[str], List[int], int, str],
    conv_codes: List[str],
    strptime_attr: str,
    name: str,
) -> np.ndarray:
    """
    Match times by applying conversion codes to filtering list.

    Parameters
    ----------
    data
        Input data to perform filtering on

    times
        Times to match

    conv_codes
        If :obj:`times` contains strings, conversion codes to try passing to
        :func:`time.strptime` to convert :obj:`times` to :class:`datetime.datetime`

    strptime_attr
        If :obj:`times` contains strings, the :class:`datetime.datetime` attribute to
        finalize the conversion of strings to integers

    name
        Name of the part of a datetime to extract, used to produce useful error
        messages.

    Returns
    -------
    :obj:`np.array` of :obj:`bool`
        Array where ``True`` indicates a match

    Raises
    ------
    ValueError
        If input times cannot be converted understood or if input strings do not lead to
        increasing integers (i.e. "Nov-Feb" will not work, one must use ["Nov-Dec",
        "Jan-Feb"] instead)
    """
    times_list = [times] if isinstance(times, (int, str)) else times

    def conv_strs(strs_to_convert, conv_codes, name):
        res = None
        for conv_code in conv_codes:
            try:
                res = [
                    getattr(time.strptime(t, conv_code), strptime_attr)
                    for t in strs_to_convert
                ]
                break
            except ValueError:
                continue

        if res is None:
            error_msg = "Could not convert {} '{}' to integer".format(
                name, strs_to_convert
            )
            raise ValueError(error_msg)
        return res

    if isinstance(times_list[0], str):
        to_delete = []
        to_append = []  # type: List
        for i, timeset in enumerate(times_list):
            # ignore type as already established we're looking at strings
            if "-" in timeset:  # type: ignore
                ints = conv_strs(timeset.split("-"), conv_codes, name)  # type: ignore
                if ints[0] > ints[1]:
                    error_msg = (
                        "string ranges must lead to increasing integer ranges,"
                        " {} becomes {}".format(timeset, ints)
                    )
                    raise ValueError(error_msg)

                # + 1 to include last month
                to_append += [j for j in range(ints[0], ints[1] + 1)]
                to_delete.append(i)

        for i in to_delete:
            del times_list[i]

        times_list = conv_strs(times_list, conv_codes, name)
        times_list += to_append

    return is_in(data, times_list)


def datetime_match(
    data: List, dts: Union[List[datetime.datetime], datetime.datetime]
) -> np.ndarray:
    """
    Match datetimes in time columns for data filtering.

    Parameters
    ----------
    data
        Input data to perform filtering on

    dts
        Datetimes to use for filtering

    Returns
    -------
    :obj:`np.array` of :obj:`bool`
        Array where ``True`` indicates a match

    Raises
    ------
    TypeError
        :obj:`dts` contains :class:`int`
    """
    dts = [dts] if isinstance(dts, datetime.datetime) else dts
    if isinstance(dts, int) or any([isinstance(d, int) for d in dts]):
        error_msg = "`time` can only be filtered with datetimes or lists of datetimes"
        raise TypeError(error_msg)
    return is_in(data, dts)
