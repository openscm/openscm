from os.path import join

import pandas as pd
import pytest

from conftest import TEST_DATA
from openscm.parameters import ParameterType
from openscm.scmdataframe.parameter_type import guess_parameter_type


def read_magicc7_variables():
    df = pd.read_csv(join(TEST_DATA, "magicc7_parameter_types.txt"), delimiter=",")

    return df.to_dict("records")


@pytest.fixture(scope="module", params=(read_magicc7_variables()))
def magicc7_variable(request):
    yield request.param


def test_with_units(magicc7_variable):
    obs = guess_parameter_type(magicc7_variable["name"], magicc7_variable["unit"])

    exp = (
        ParameterType.AVERAGE_TIMESERIES
        if magicc7_variable["contains_time_dim"]
        else ParameterType.POINT_TIMESERIES
    )
    assert obs == exp


def test_without_units(magicc7_variable):
    obs = guess_parameter_type(magicc7_variable["name"], "")

    exp = (
        ParameterType.AVERAGE_TIMESERIES
        if magicc7_variable["contains_time_dim"]
        else ParameterType.POINT_TIMESERIES
    )
    assert obs == exp


@pytest.mark.parametrize(
    ["variable", "expected"],
    (
        ("ALLGHGS_GWPEMIS", ParameterType.AVERAGE_TIMESERIES),
        ("ALLGHGS_SRF", ParameterType.AVERAGE_TIMESERIES),
        ("Atmospheric Concentrations|C2F6", ParameterType.POINT_TIMESERIES),
        ("Emissions|HFC245fa", ParameterType.AVERAGE_TIMESERIES),
    ),
)
def test_guess(variable, expected):
    assert guess_parameter_type(variable, "") == expected


def test_rcp26_emis(rcp26):
    for v in rcp26["variable"].unique():
        assert "Emissions" in v
        assert guess_parameter_type(v, "") == ParameterType.AVERAGE_TIMESERIES
