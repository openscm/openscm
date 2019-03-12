from collections import namedtuple
from datetime import datetime


import pytest
import numpy as np
import pandas as pd


from openscm.core import ParameterSet
from openscm import timeseries_converter
from openscm.parameters import ParameterType


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
        datetime(1005, 1, 1),
        datetime(3010, 12, 31),
    ],
)


@pytest.fixture(scope="function")
def test_pd_df():
    yield TEST_DF


@pytest.fixture(scope="function")
def test_adapter(request):
    """
    Get an initialized instance of an the requesting classes ``tadapter`` property.
    """
    parameters = ParameterSet()
    parameters.get_writable_scalar_view("ecs", ("World",), "K").set(3)
    parameters.get_writable_scalar_view("rf2xco2", ("World",), "W / m^2").set(4.0)
    output_parameters = ParameterSet()
    try:
        yield request.cls.tadapter(parameters, output_parameters)
    except TypeError:
        pytest.skip("{} cannot be instantiated".format(str(request.cls.tadapter)))


@pytest.fixture(scope="function")
def test_run_parameters():

    run_parameters = namedtuple("RunParameters", ["start_time", "stop_time"])
    run_parameters.start_time = 0
    run_parameters.stop_time = 1
    yield run_parameters


possible_source_values = [[1, 5, 3, 5, 7, 3, 2, 9]]

possible_target_values = [
    dict(
        source_start_time=0,
        source_period_length=10,
        target_start_time=-5,
        target_period_length=5,
        source_values=possible_source_values[0],
        target_values=[
            -1,
            1,
            3,
            5,
            4,
            3,
            4,
            5,
            6,
            7,
            5,
            3,
            2.5,
            2,
            5.5,
            9,
            12.5,
        ],
        timeseries_type=ParameterType.POINT_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.LINEAR,
    ),
    dict(
        source_start_time=0,
        source_period_length=10,
        target_start_time=-50,
        target_period_length=50,
        source_values=possible_source_values[0],
        target_values=[
            1,
            1,
            3,
            9,
        ],
        timeseries_type=ParameterType.POINT_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.CONSTANT,
    ),
    dict(
        source_start_time=0,
        source_period_length=10,
        target_start_time=-50,
        target_period_length=50,
        source_values=possible_source_values[0],
        target_values=[
            1,
            1,
            3,
            9,
        ],
        timeseries_type=ParameterType.POINT_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.NONE,
        # check this raises ValueError with useful error message somehow
    ),
    dict(
        source_start_time=0,
        source_period_length=10,
        target_start_time=4,
        target_period_length=7,
        source_values=possible_source_values[0],
        target_values=[
            2.02142857,
            5.37857143,
            3.42857143,
            3.34285714,
            5.1,
            7.18571429,
            5.44642857,
            2.49642857,
            1.20714286,
            5.59285714,
            10.75,
        ],
        timeseries_type=ParameterType.AVERAGE_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.LINEAR,
    ),
    dict(
        source_start_time=0,
        source_period_length=10,
        target_start_time=0,
        target_period_length=5,
        source_values=possible_source_values[0],
        target_values=[
            0.0,
            2.0,
            4.75,
            5.25,
            3.0,
            3.0,
            4.5,
            5.5,
            7.25,
            6.75,
            3.625,
            2.375,
            1.25,
            2.75,
            7.25,
            10.75,
        ],
        timeseries_type=ParameterType.AVERAGE_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.LINEAR,
    ),
    dict(
        source_start_time=3,
        source_period_length=3,
        target_start_time=0,
        target_period_length=5,
        source_values=possible_source_values[0],
        target_values=[-1.66666667, 4.13333333, 4.13333333, 5.51666667, 3.01666667],
        timeseries_type=ParameterType.AVERAGE_TIMESERIES,
        interpolation_type=timeseries_converter.InterpolationType.LINEAR,
        extrapolation_type=timeseries_converter.ExtrapolationType.LINEAR,
    ),
]

test_combinations = []

for index in possible_target_values:
    combination = namedtuple(
        "TestCombination",
        [
            "source",
            "source_values",
            "target",
            "target_values",
            "timeseries_type",
            "interpolation_type",
            "extrapolation_type",
        ],
    )
    combination.source_values = np.array(index["source_values"])
    combination.source = timeseries_converter.create_time_points(
        index["source_start_time"],
        index["source_period_length"],
        len(combination.source_values),
        ParameterType.AVERAGE_TIMESERIES,
    )
    combination.target_values = np.array(index["target_values"])
    combination.target = timeseries_converter.create_time_points(
        index["target_start_time"],
        index["target_period_length"],
        len(combination.target_values),
        ParameterType.AVERAGE_TIMESERIES,
    )
    combination.timeseries_type = index["timeseries_type"]
    combination.interpolation_type = index["interpolation_type"]
    combination.extrapolation_type = index["extrapolation_type"]
    test_combinations.append(combination)


@pytest.fixture(params=test_combinations)
def combo(request):
    return request.param
