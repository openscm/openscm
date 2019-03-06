from collections import namedtuple
import numpy as np
import pytest


from openscm import timeseries_converter
from openscm.errors import InsufficientDataError
from openscm.parameters import ParameterType


possible_source_values = [[1, 5, 3, 5, 7, 3, 2, 9]]

possible_target_values = [
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


def test_conversion_to_same_timeseries(combo):
    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.source,
        combo.source,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter._convert(
        combo.source_values, combo.source, combo.source
    )
    np.testing.assert_allclose(values, combo.source_values)
    assert timeseriesconverter.source_length == len(combo.source) - (
        1 if combo.timeseries_type == ParameterType.AVERAGE_TIMESERIES else 0
    )
    assert timeseriesconverter.target_length == len(combo.source) - (
        1 if combo.timeseries_type == ParameterType.AVERAGE_TIMESERIES else 0
    )


def test_insufficient_overlap(combo):
    with pytest.raises(InsufficientDataError):
        timeseries_converter.TimeseriesConverter(
            combo.source,
            combo.target - 1e6,
            combo.timeseries_type,
            combo.interpolation_type,
            combo.extrapolation_type,
        )


def test_short_data(combo):
    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.source,
        combo.target,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    for a in [[], [0], [0, 1]]:
        with pytest.raises(InsufficientDataError):
            timeseriesconverter._convert(np.array(a), combo.source, combo.target)


def test_conversion(combo):
    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.source,
        combo.target,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter._convert(
        combo.source_values, combo.source, combo.target
    )
    np.testing.assert_allclose(values, combo.target_values, atol=1e-10 * values.max())


def test_timeseriesconverter(combo):
    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.source,
        combo.target,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter.convert_from(combo.source_values)
    np.testing.assert_allclose(values, combo.target_values, atol=1e-10 * values.max())

    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.target,
        combo.source,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter.convert_to(combo.source_values)
    np.testing.assert_allclose(values, combo.target_values, atol=1e-10 * values.max())
