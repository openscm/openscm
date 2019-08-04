import re

import numpy as np
import pytest

from openscm.core.parameters import ParameterType
from openscm.core.time import ExtrapolationType, InterpolationType, TimeseriesConverter
from openscm.errors import InsufficientDataError


def test_short_data(combo):
    timeseriesconverter = TimeseriesConverter(
        combo.source,
        combo.target,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    for a in [[], [0], [0, 1]]:
        with pytest.raises(InsufficientDataError):
            timeseriesconverter._convert(np.array(a), combo.source, combo.target)


def test_none_extrapolation_error(combo):
    target = np.asarray(
        [
            combo.source[0] - np.timedelta64(1, "s"),
            combo.source[0],
            combo.source[-1] + np.timedelta64(1, "s"),
        ],
        dtype=np.datetime64,
    )
    error_msg = re.escape(
        "Target time points are outside the source time points, use an "
        "extrapolation type other than None"
    )
    with pytest.raises(InsufficientDataError, match=error_msg):
        TimeseriesConverter(
            combo.source,
            target,
            combo.timeseries_type,
            combo.interpolation_type,
            ExtrapolationType.NONE,
        )


@pytest.mark.parametrize("miss_type", ["forward", "backward"])
def test_no_overlap(combo, miss_type):
    if miss_type == "forward":
        missing_overlap_target = combo.source.copy()
        missing_overlap_target[0] -= np.timedelta64(1, "D")
    else:
        missing_overlap_target = combo.source.copy()
        missing_overlap_target[-1] += np.timedelta64(1, "D")

    with pytest.raises(InsufficientDataError):
        TimeseriesConverter(
            combo.source,
            missing_overlap_target,
            combo.timeseries_type,
            combo.interpolation_type,
            ExtrapolationType.NONE,
        )

    # all ok if extrapolation is not NONE
    TimeseriesConverter(
        combo.source,
        missing_overlap_target,
        combo.timeseries_type,
        combo.interpolation_type,
        ExtrapolationType.LINEAR,
    )


def test_point_to_average_conversion():
    point_times = np.array([  0, 1, 2, 5, 10])
    point_values = np.array([-1, 2, 3, 6, 5.5])

    average_times = np.array([0, 2, 3, 10])
    average_values = np.array([1, 2.5, 6.5])

    converter_point_to_average = TimeseriesConverter(
        point_times,
        average_times,
        ParameterType.POINT_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.NONE,
        timeseries_type_target=ParameterType.AVERAGE_TIMESERIES,
    )
    converter_average_to_point_circular = TimeseriesConverter(
        average_times,
        point_times,
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.NONE,
        timeseries_type_target=ParameterType.POINT_TIMESERIES,
    )

    point_to_average_result = converter_point_to_average.convert_from(point_values)
    point_to_average_expected = np.array([1.5, 3.5, 6.25])
    np.testing.assert_array_equal(point_to_average_result, point_to_average_expected)

    average_to_point_result_circular = converter_average_to_point_circular.convert_from(
        point_to_average_result
    )
    np.testing.assert_array_equal(average_to_point_result_circular, point_values)

    converter_average_to_point = TimeseriesConverter(
        average_times,
        np.arange(0, 11),
        ParameterType.AVERAGE_TIMESERIES,
        InterpolationType.LINEAR,
        ExtrapolationType.NONE,
        timeseries_type_target=ParameterType.POINT_TIMESERIES,
    )
    average_to_point_result = converter_average_to_point.convert_from(
        average_values
    )
    np.testing.assert_array_equal(average_to_point_result, np.arange(0, 11))
