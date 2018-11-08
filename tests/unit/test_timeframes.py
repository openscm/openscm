import numpy as np
from openscm import timeframes
import pytest


@pytest.fixture(params=[(0, 10), (5, 100)])
def source(request):
    return timeframes.Timeframe(
        start_time=request.param[0], period_length=request.param[1]
    )


@pytest.fixture(params=[(0, 7), (17, 83)])
def target(request):
    return timeframes.Timeframe(
        start_time=request.param[0], period_length=request.param[1]
    )


@pytest.fixture(params=[0])
def source_values(request):
    return [np.array([1, 5, 3, 5, 7, 3, 2, 9])][request.param]


def test_conversion_to_same_timeframe(source, source_values):
    target_values = timeframes._convert(source_values, source, source)
    np.testing.assert_array_equal(target_values, source_values)
