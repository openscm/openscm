import numpy as np
import pytest

from openscm.core.parameters import ParameterType
from openscm.core.time import TimeseriesConverter, ExtrapolationType
from openscm.errors import InsufficientDataError


def test_conversion_to_same_timeseries(combo):
    timeseriesconverter = TimeseriesConverter(
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
        TimeseriesConverter(
            combo.source,
            combo.target - np.timedelta64(3650, "D"),
            combo.timeseries_type,
            combo.interpolation_type,
            ExtrapolationType.NONE,
        )


def test_conversion(combo):
    timeseriesconverter = TimeseriesConverter(
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
    timeseriesconverter = TimeseriesConverter(
        combo.source,
        combo.target,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter.convert_from(combo.source_values)
    np.testing.assert_allclose(values, combo.target_values, atol=1e-10 * values.max())

    timeseriesconverter = TimeseriesConverter(
        combo.target,
        combo.source,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    values = timeseriesconverter.convert_to(combo.source_values)
    np.testing.assert_allclose(values, combo.target_values, atol=1e-10 * values.max())
