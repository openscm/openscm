from datetime import datetime


import pytest
import numpy as np
import pandas as pd
from pyam import IamDataFrame


from openscm.highlevel import ScmDataFrame


TEST_DF_LONG_TIMES = pd.DataFrame(
    [
        ["a_model", "a_iam", "a_scenario", "World", "Primary Energy", "EJ/y", 1, 6.0],
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy|Coal",
            "EJ/y",
            0.5,
            3,
        ],
        ["a_model", "a_iam", "a_scenario2", "World", "Primary Energy", "EJ/y", 2, 7],
    ],
    columns=[
        "climate_model",
        "model",
        "scenario",
        "region",
        "variable",
        "unit",
        datetime(1005, 1, 1),
        datetime(3010, 12, 31),
    ],
)

TEST_DF = pd.DataFrame(
    [
        ["a_model", "a_iam", "a_scenario", "World", "Primary Energy", "EJ/y", 1, 6.0],
        [
            "a_model",
            "a_iam",
            "a_scenario",
            "World",
            "Primary Energy|Coal",
            "EJ/y",
            0.5,
            3,
        ],
        ["a_model", "a_iam", "a_scenario2", "World", "Primary Energy", "EJ/y", 2, 7],
    ],
    columns=[
        "climate_model",
        "model",
        "scenario",
        "region",
        "variable",
        "unit",
        2005,
        2010,
    ],
)

TEST_TS = np.array([[1, 6.0], [0.5, 3], [2, 7]]).T


@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF


@pytest.fixture(scope="function")
def test_scm_datetime_df():
    tdf = TEST_DF.copy()
    tdf.rename(
        {2005: datetime(2005, 6, 17, 12), 2010: datetime(2010, 1, 3, 0)},
        axis="columns",
        inplace=True,
    )

    yield ScmDataFrame(tdf)


@pytest.fixture(scope="function")
def test_pd_longtime_df():
    yield TEST_DF_LONG_TIMES


@pytest.fixture(scope="function")
def test_ts():
    yield TEST_TS


@pytest.fixture(scope="function")
def test_iam_df():
    yield IamDataFrame(TEST_DF)


@pytest.fixture(
    scope="function",
    params=[
        {"data": TEST_DF},
        {"data": IamDataFrame(TEST_DF).data},
        {"data": IamDataFrame(TEST_DF).timeseries()},
        {
            "data": TEST_TS,
            "columns": {
                "index": [2005, 2010],
                "model": ["a_iam"],
                "climate_model": ["a_model"],
                "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
                "region": ["World"],
                "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
                "unit": ["EJ/y"],
            },
        },
    ],
)
def test_scm_df(request):
    yield ScmDataFrame(**request.param)


def test_adapter(request):
    return request.cls.tadapter()


def assert_core(
    expected, time, test_core, name, region, unit, start, period_length
):
    pview = test_core.parameters.get_timeseries_view(name, region, unit, start, period_length)
    relevant_idx = (np.abs(pview.get_times() - time)).argmin()
    np.testing.assert_allclose(pview.get(relevant_idx), expected)


# temporary workaround until this is in Pint itself and can be imported
def assert_pint_equal(a, b, **kwargs):
    c = b.to(a.units)
    try:
        np.testing.assert_array_equal(a.magnitude, c.magnitude, **kwargs)

    except AssertionError as e:
        original_msg = "{}".format(e)
        note_line = "Note: values above have been converted to {}".format(a.units)
        units_lines = "Input units:\n" "x: {}\n" "y: {}".format(a.units, b.units)

        numerical_lines = (
            "Numerical values with units:\n" "x: {}\n" "y: {}".format(a, b)
        )

        error_msg = (
            "{}\n"
            "\n"
            "{}\n"
            "\n"
            "{}\n"
            "\n"
            "{}".format(original_msg, note_line, units_lines, numerical_lines)
        )

        raise AssertionError(error_msg)
