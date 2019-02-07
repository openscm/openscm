import datetime
import warnings
from importlib import reload


import pytest
import numpy as np


import openscm.utils
from openscm.utils import (
    convert_datetime_to_openscm_time,
    convert_openscm_time_to_datetime,
)


@pytest.fixture(scope="function")
def ensure_input_is_tuple_instance():
    # fresh import each time for counting number of calls
    reload(openscm.utils)
    yield openscm.utils.ensure_input_is_tuple


@pytest.fixture(scope="module")
def warn_message():
    yield "Converting input {} from string to tuple"


def test_convert_string_to_tuple_tuple_input(ensure_input_is_tuple_instance):
    with warnings.catch_warnings(record=True) as warn_tuple_input:
        ensure_input_is_tuple_instance(("test",))

    assert len(warn_tuple_input) == 0


def test_convert_string_to_tuple_tuple_forgot_comma(
    ensure_input_is_tuple_instance, warn_message
):
    tinp = "test"
    with warnings.catch_warnings(record=True) as warn_tuple_forgot_comma_input:
        ensure_input_is_tuple_instance(tinp)

    assert len(warn_tuple_forgot_comma_input) == 1
    assert str(warn_tuple_forgot_comma_input[0].message) == warn_message.format(tinp)


def test_convert_string_to_tuple_list_input(
    ensure_input_is_tuple_instance, warn_message
):
    with warnings.catch_warnings(record=True) as warn_list_input:
        ensure_input_is_tuple_instance(["test"])

    assert len(warn_list_input) == 0


def test_convert_string_to_tuple_str_input(
    ensure_input_is_tuple_instance, warn_message
):
    tinp1 = "test"
    with warnings.catch_warnings(record=True) as warn_tuple_str_input:
        ensure_input_is_tuple_instance(tinp1)
        ensure_input_is_tuple_instance("test 2")

    assert len(warn_tuple_str_input) == 1  # make sure only thrown once
    assert str(warn_tuple_str_input[0].message) == warn_message.format(tinp1)


@pytest.mark.parametrize(
    "input, expected",
    [
        (datetime.datetime(1970, 1, 1, 0, 0, 0), 0),
        (datetime.datetime(1970, 1, 1, 0, 1, 0), 60),
        (datetime.datetime(1970, 1, 1, 0, 1, 3), 63),
        (datetime.datetime(1969, 12, 31, 23, 50, 3), -597),
        (datetime.datetime(1880, 1, 2, 0, 0, 0), -2840054400),
        (datetime.datetime(2070, 1, 1, 0, 0, 0), 3155760000),
    ],
)
def test_convert_datetime_to_openscm_time(input, expected):
    res = convert_datetime_to_openscm_time(input)

    np.testing.assert_allclose(res, expected)
    assert convert_openscm_time_to_datetime(res) == input
