import datetime as dt
import re

import numpy as np
import numpy.testing as npt
import pytest

from openscm.scmdataframe.timeindex import TimeIndex, to_int


def test_to_int_value_error():
    error_msg = re.escape("invalid values `{}`".format([4.5, 6.5]))
    with pytest.raises(ValueError, match=error_msg):
        to_int(np.array([1, 3, 4.5, 6.5, 7.0, 8]))


def test_to_int_type_error():
    inp = [1, 3, 4.5, 6.5, 7.0, 8]
    error_msg = re.escape(
        "For our own sanity, this method only works with np.ndarray input. x is "
        "type: {}".format(type(inp))
    )
    with pytest.raises(TypeError, match=error_msg):
        to_int(inp)


def test_timeindex_init_error():
    error_msg = re.escape("One of `py_dt` or `openscm_dt` must be supplied")
    with pytest.raises(AssertionError, match=error_msg):
        TimeIndex()


def test_timeindex_init_py_dt_error():
    inp = ["junk", "here"]
    error_msg = re.escape(
        "All time values must be convertible to datetime. The "
        "following values are not:\n\t{}".format(np.asarray(inp))
    )
    with pytest.raises(ValueError, match=error_msg):
        TimeIndex(py_dt=inp)


def test_timeindex_init_openscm_dt():
    inp = [0, 10]
    idx = TimeIndex(openscm_dt=inp)
    npt.assert_array_equal(
        idx.as_py(), [dt.datetime(1970, 1, 1), dt.datetime(1970, 1, 1, 0, 0, 10)]
    )


def test_timeindex_immutable():
    inp = [0, 10]
    idx = TimeIndex(openscm_dt=inp)
    error_msg = re.escape("TimeIndex is immutable")
    with pytest.raises(AttributeError, match=error_msg):
        idx._openscm = inp


def test_timeindex_item_assignment():
    inp = [0, 10]
    idx = TimeIndex(openscm_dt=inp)
    error_msg = re.escape("'TimeIndex' object does not support item assignment")
    with pytest.raises(TypeError, match=error_msg):
        idx["openscm"] = inp
