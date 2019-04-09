"""
Helpers for filtering DataFrames

Borrowed from pyam.utils
"""

import collections
import datetime
import re
import time

import numpy as np
import six


def is_str(s):
    return isinstance(s, six.string_types)


def find_depth(data, s, level):
    # determine function for finding depth level =, >=, <= |s
    if not is_str(level):
        test = lambda x: level == x
    elif level[-1] == '-':
        level = int(level[:-1])
        test = lambda x: level >= x
    elif level[-1] == '+':
        level = int(level[:-1])
        test = lambda x: level <= x
    else:
        raise ValueError('Unknown level type: {}'.format(level))

    # determine depth
    pipe = re.compile('\\|')
    regexp = str(s).replace('*', '')
    apply_test = lambda val: test(len(pipe.findall(val.replace(regexp, ''))))
    return list(map(apply_test, data))


def pattern_match(data, values, level=None, regexp=False, has_nan=True):
    """
    matching of model/scenario names, variables, regions, and meta columns to
    pseudo-regex (if `regexp == False`) for filtering (str, int, bool)
    """
    matches = np.array([False] * len(data))
    if not isinstance(values, collections.Iterable) or is_str(values):
        values = [values]

    # issue (#40) with string-to-nan comparison, replace nan by empty string
    _data = data.copy()
    if has_nan:
        _data.loc[[np.isnan(i) if not is_str(i) else False for i in _data]] = ''

    for s in values:
        if is_str(s):
            _regexp = (str(s)
                       .replace('|', '\\|')
                       .replace('.', '\.')  # `.` has to be replaced before `*`
                       .replace('*', '.*')
                       .replace('+', '\+')
                       .replace('(', '\(')
                       .replace(')', '\)')
                       .replace('$', '\\$')
                       ) + "$"
            pattern = re.compile(_regexp if not regexp else s)

            subset = filter(pattern.match, _data)
            depth = True if level is None else find_depth(_data, s, level)
            matches |= (_data.isin(subset) & depth)
        else:
            matches |= data == s
    return matches


def years_match(data, years):
    """
    matching of year columns for data filtering
    """
    years = [years] if isinstance(years, int) else years
    dt = datetime.datetime
    if isinstance(years, dt) or isinstance(years[0], dt):
        error_msg = "`year` can only be filtered with ints or lists of ints"
        raise TypeError(error_msg)
    return data.isin(years)


def month_match(data, months):
    """
    matching of months in time columns for data filtering
    """
    return time_match(data, months, ['%b', '%B'], "tm_mon", "months")


def day_match(data, days):
    """
    matching of days in time columns for data filtering
    """
    return time_match(data, days, ['%a', '%A'], "tm_wday", "days")


def hour_match(data, hours):
    """
    matching of days in time columns for data filtering
    """
    hours = [hours] if isinstance(hours, int) else hours
    return data.isin(hours)


def time_match(data, times, conv_codes, strptime_attr, name):
    def conv_strs(strs_to_convert, conv_codes, name):
        res = None
        for conv_code in conv_codes:
            try:
                res = [getattr(time.strptime(t, conv_code), strptime_attr)
                       for t in strs_to_convert]
                break
            except ValueError:
                continue

        if res is None:
            raise ValueError("Could not convert {} to integer".format(name))
        return res

    times = [times] if isinstance(times, (int, str)) else times
    if isinstance(times[0], str):
        to_delete = []
        to_append = []
        for i, timeset in enumerate(times):
            if "-" in timeset:
                ints = conv_strs(timeset.split("-"), conv_codes, name)
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
            del times[i]

        times = conv_strs(times, conv_codes, name)
        times += to_append

    return data.isin(times)


def datetime_match(data, dts):
    """
    matching of datetimes in time columns for data filtering
    """
    dts = [dts] if isinstance(dts, datetime.datetime) else dts
    if isinstance(dts, int) or isinstance(dts[0], int):
        error_msg = (
            "`time` can only be filtered with datetimes or lists of datetimes"
        )
        raise TypeError(error_msg)
    return data.isin(dts)
